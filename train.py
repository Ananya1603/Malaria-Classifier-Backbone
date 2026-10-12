import argparse
import csv
import json
import math
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import build_dataset, build_validation_dataset
from models import build_model


CHECKPOINTS = {
    "dinov2": "Sadou/malaria-detector-dinov2-tanzania",
}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_logits(model, images):
    output = model(images)

    if hasattr(output, "logits"):
        output = output.logits

    return output.reshape(-1)


def calculate_pos_weight(dataset):
    """
    Project labels:
        0 = Parasitized
        1 = Uninfected

    BCEWithLogitsLoss treats label 1 as positive.

    Required BBBC041 weighting:
        pos_weight =
        number_of_negative_training_examples /
        number_of_positive_training_examples

    This is calculated ONLY from the actual training subset.
    """
    if not (
        hasattr(dataset, "filenames")
        and hasattr(dataset, "label_map")
    ):
        return None

    negatives = 0
    positives = 0

    for filename in dataset.filenames:
        label = int(dataset.label_map[filename])

        if label == 1:
            positives += 1
        elif label == 0:
            negatives += 1
        else:
            raise RuntimeError(
                f"Unexpected label: {label}"
            )

    if positives == 0:
        raise RuntimeError(
            "Cannot calculate pos_weight: "
            "no positive examples."
        )

    return negatives / positives


@torch.no_grad()
def evaluate(
    model,
    loader,
    criterion,
    device,
):
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for images, labels in loader:
        images = images.to(
            device,
            non_blocking=True,
        )

        labels = labels.to(
            device,
            non_blocking=True,
        ).reshape(-1)

        # AMP is also used for validation inference.
        with torch.amp.autocast(
            device_type="cuda",
            dtype=torch.float16,
        ):
            logits = get_logits(
                model,
                images,
            )

            loss = criterion(
                logits,
                labels,
            )

        batch_size = labels.numel()

        total_loss += (
            loss.item() * batch_size
        )

        predictions = (
            torch.sigmoid(logits) >= 0.5
        ).float()

        total_correct += (
            predictions == labels
        ).sum().item()

        total_samples += batch_size

    if total_samples == 0:
        raise RuntimeError(
            "Validation dataset is empty."
        )

    validation_loss = (
        total_loss / total_samples
    )

    validation_accuracy = (
        total_correct / total_samples
    )

    model.train()

    return (
        validation_loss,
        validation_accuracy,
    )


def save_accuracy_graph(
    history,
    output_path,
):
    if not history:
        return

    epochs = [
        row["equivalent_epoch"]
        for row in history
    ]

    train_accuracy = [
        row["train_accuracy"]
        for row in history
    ]

    validation_accuracy = [
        row["validation_accuracy"]
        for row in history
    ]

    plt.figure(figsize=(9, 6))

    plt.plot(
        epochs,
        train_accuracy,
        label="Training Accuracy",
    )

    plt.plot(
        epochs,
        validation_accuracy,
        label="Validation Accuracy",
    )

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(
        "Training vs Validation Accuracy"
    )
    plt.ylim(0.0, 1.0)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=200,
    )

    plt.close()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        required=True,
        choices=["dinov2"],
    )

    parser.add_argument(
        "--dataset",
        required=True,
        choices=[
            "nih",
            "bbbc041",
        ],
    )

    parser.add_argument(
        "--fraction",
        required=True,
        type=int,
        choices=[
            100,
            25,
            10,
            2,
        ],
    )

    parser.add_argument(
        "--seed",
        required=True,
        type=int,
        choices=[
            42,
            43,
        ],
    )

    parser.add_argument(
        "--max-steps",
        type=int,
        default=5000,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=1e-5,
    )

    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.01,
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--eval-every",
        type=int,
        default=250,
    )

    args = parser.parse_args()

    if args.batch_size != 32:
        raise ValueError(
            "Official Milestone 4 protocol requires "
            "batch size exactly 32."
        )

    if args.max_steps <= 0:
        raise ValueError(
            "max_steps must be positive."
        )

    if args.eval_every <= 0:
        raise ValueError(
            "eval_every must be positive."
        )

    set_seed(args.seed)

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. "
            "Do not run DINOv2 on CPU."
        )

    device = torch.device("cuda")

    print("=" * 60)
    print("MILESTONE 4 â€” DINOv2")
    print("=" * 60)

    print("Model:", args.model)
    print(
        "Checkpoint:",
        CHECKPOINTS[args.model],
    )
    print("Dataset:", args.dataset)
    print("Fraction:", args.fraction)
    print("Training seed:", args.seed)
    print(
        "Maximum optimizer steps:",
        args.max_steps,
    )
    print("Batch size:", args.batch_size)
    print("Learning rate:", args.lr)
    print("Weight decay:", args.weight_decay)
    print(
        "Evaluation every steps:",
        args.eval_every,
    )
    print("Input size: 518 x 518")
    print("Mixed precision (AMP): enabled")
    print(
        "GPU:",
        torch.cuda.get_device_name(0),
    )

    print("\nLoading training dataset...")

    train_dataset = build_dataset(
        dataset_name=args.dataset,
        fraction=args.fraction,
        seed=args.seed,
        model_name=args.model,
    )

    print(
        "Training samples:",
        len(train_dataset),
    )

    print(
        "\nLoading FIXED validation dataset..."
    )

    validation_dataset = (
        build_validation_dataset(
            dataset_name=args.dataset,
            model_name=args.model,
        )
    )

    print(
        "Validation samples:",
        len(validation_dataset),
    )

    generator = torch.Generator()
    generator.manual_seed(args.seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        generator=generator,
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    steps_per_epoch = math.ceil(
        len(train_dataset)
        / args.batch_size
    )

    print(
        "Steps per equivalent epoch:",
        steps_per_epoch,
    )

    print(
        "\nLoading DINOv2 checkpoint..."
    )

    model = build_model(
        args.model
    ).to(device)

    # Explicitly ensure the complete backbone
    # and fresh classifier are trainable.
    for parameter in model.parameters():
        parameter.requires_grad = True

    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    total_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    print(
        "Trainable parameters:",
        trainable_parameters,
    )

    print(
        "Total parameters:",
        total_parameters,
    )

    if (
        trainable_parameters
        != total_parameters
    ):
        raise RuntimeError(
            "The entire DINOv2 model "
            "must be trainable."
        )

    pos_weight_value = None
    pos_weight_tensor = None

    if args.dataset == "bbbc041":
        pos_weight_value = (
            calculate_pos_weight(
                train_dataset
            )
        )

        pos_weight_tensor = torch.tensor(
            [pos_weight_value],
            dtype=torch.float32,
            device=device,
        )

        print(
            "BBBC041 training-subset "
            "pos_weight:",
            pos_weight_value,
        )

    criterion = (
        torch.nn.BCEWithLogitsLoss(
            pos_weight=pos_weight_tensor
        )
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # AMP gradient scaler.
    scaler = torch.amp.GradScaler(
        "cuda"
    )

    run_name = (
        f"{args.model}_"
        f"{args.dataset}_"
        f"fraction{args.fraction}_"
        f"seed{args.seed}"
    )

    output_dir = (
        Path("outputs")
        / run_name
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    log_path = (
        output_dir
        / "training_log.csv"
    )

    config_path = (
        output_dir
        / "config.json"
    )

    graph_path = (
        output_dir
        / "accuracy_curve.png"
    )

    final_checkpoint_path = (
        output_dir
        / "final_model.pt"
    )

    best_checkpoint_path = (
        output_dir
        / "best_validation_model.pt"
    )

    config = {
        "model": args.model,
        "checkpoint": CHECKPOINTS[
            args.model
        ],
        "dataset": args.dataset,
        "fraction_percent":
            args.fraction,
        "training_seed": args.seed,
        "max_optimizer_steps":
            args.max_steps,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "weight_decay":
            args.weight_decay,
        "optimizer": "AdamW",
        "loss": "BCEWithLogitsLoss",
        "pos_weight":
            pos_weight_value,
        "training_samples": len(
            train_dataset
        ),
        "validation_samples": len(
            validation_dataset
        ),
        "validation_fixed": True,
        "test_used_during_training":
            False,
        "input_size": [
            518,
            518,
        ],
        "normalization_mean": [
            0.485,
            0.456,
            0.406,
        ],
        "normalization_std": [
            0.229,
            0.224,
            0.225,
        ],
        "checkpoint_native_processor_size":
            224,
        "experiment_input_size_override":
            518,
        "eval_every_optimizer_steps":
            args.eval_every,
        "full_backbone_trainable":
            True,
        "fresh_binary_head": True,
        "output_logits": 1,
        "stopping_rule":
            "exact_optimizer_step_count",
        "mixed_precision": True,
        "amp_dtype": "float16",
    }

    with open(
        config_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            config,
            f,
            indent=2,
        )

    with open(
        log_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.writer(f)

        writer.writerow([
            "optimizer_step",
            "equivalent_epoch",
            "train_loss",
            "train_accuracy",
            "validation_loss",
            "validation_accuracy",
        ])

    optimizer_step = 0

    interval_loss_sum = 0.0
    interval_correct = 0
    interval_samples = 0

    history = []

    best_validation_loss = float(
        "inf"
    )

    model.train()

    print()
    print("=" * 60)
    print("STARTING TRAINING")
    print("=" * 60)

    while optimizer_step < args.max_steps:

        for images, labels in train_loader:

            if (
                optimizer_step
                >= args.max_steps
            ):
                break

            images = images.to(
                device,
                non_blocking=True,
            )

            labels = labels.to(
                device,
                non_blocking=True,
            ).reshape(-1)

            optimizer.zero_grad(
                set_to_none=True
            )

            # Mixed-precision forward pass.
            with torch.amp.autocast(
                device_type="cuda",
                dtype=torch.float16,
            ):
                logits = get_logits(
                    model,
                    images,
                )

                loss = criterion(
                    logits,
                    labels,
                )

            # AMP backward pass + optimizer step.
            scaler.scale(
                loss
            ).backward()

            scaler.step(
                optimizer
            )

            scaler.update()

            optimizer_step += 1

            if optimizer_step % 10 == 0 or optimizer_step == 1:
                print(f"Training progress: step {optimizer_step}/{args.max_steps}", flush=True)

            with torch.no_grad():
                predictions = (
                    torch.sigmoid(logits)
                    >= 0.5
                ).float()

                correct = (
                    predictions == labels
                ).sum().item()

            batch_samples = (
                labels.numel()
            )

            interval_loss_sum += (
                loss.item()
                * batch_samples
            )

            interval_correct += correct

            interval_samples += (
                batch_samples
            )

            should_evaluate = (
                optimizer_step == 1
                or optimizer_step
                % args.eval_every == 0
                or optimizer_step
                == args.max_steps
            )

            if not should_evaluate:
                continue

            train_loss = (
                interval_loss_sum
                / interval_samples
            )

            train_accuracy = (
                interval_correct
                / interval_samples
            )

            equivalent_epoch = (
                optimizer_step
                / steps_per_epoch
            )

            print()
            print(
                f"Evaluating at step "
                f"{optimizer_step}/"
                f"{args.max_steps}..."
            )

            (
                validation_loss,
                validation_accuracy,
            ) = evaluate(
                model,
                validation_loader,
                criterion,
                device,
            )

            row = {
                "optimizer_step":
                    optimizer_step,
                "equivalent_epoch":
                    equivalent_epoch,
                "train_loss":
                    train_loss,
                "train_accuracy":
                    train_accuracy,
                "validation_loss":
                    validation_loss,
                "validation_accuracy":
                    validation_accuracy,
            }

            history.append(row)

            with open(
                log_path,
                "a",
                newline="",
                encoding="utf-8",
            ) as f:
                writer = csv.writer(f)

                writer.writerow([
                    optimizer_step,
                    equivalent_epoch,
                    train_loss,
                    train_accuracy,
                    validation_loss,
                    validation_accuracy,
                ])

            print(
                f"Step "
                f"{optimizer_step:5d}/"
                f"{args.max_steps} | "
                f"epoch "
                f"{equivalent_epoch:.3f} | "
                f"train loss "
                f"{train_loss:.4f} | "
                f"train acc "
                f"{train_accuracy:.4f} | "
                f"val loss "
                f"{validation_loss:.4f} | "
                f"val acc "
                f"{validation_accuracy:.4f}"
            )

            save_accuracy_graph(
                history,
                graph_path,
            )

            if (
                validation_loss
                < best_validation_loss
            ):
                best_validation_loss = (
                    validation_loss
                )

                torch.save(
                    {
                        "model_state_dict":
                            model.state_dict(),
                        "optimizer_step":
                            optimizer_step,
                        "validation_loss":
                            validation_loss,
                        "validation_accuracy":
                            validation_accuracy,
                        "model":
                            args.model,
                        "dataset":
                            args.dataset,
                        "fraction":
                            args.fraction,
                        "seed":
                            args.seed,
                    },
                    best_checkpoint_path,
                )

                print(
                    "Saved new best "
                    "validation checkpoint."
                )

            interval_loss_sum = 0.0
            interval_correct = 0
            interval_samples = 0

    torch.save(
        {
            "model_state_dict":
                model.state_dict(),
            "optimizer_state_dict":
                optimizer.state_dict(),
            "model":
                args.model,
            "dataset":
                args.dataset,
            "fraction":
                args.fraction,
            "seed":
                args.seed,
            "optimizer_steps":
                optimizer_step,
            "batch_size":
                args.batch_size,
            "learning_rate":
                args.lr,
            "weight_decay":
                args.weight_decay,
            "pos_weight":
                pos_weight_value,
            "mixed_precision":
                True,
            "amp_dtype":
                "float16",
        },
        final_checkpoint_path,
    )

    config[
        "completed_optimizer_steps"
    ] = optimizer_step

    config[
        "best_validation_loss"
    ] = best_validation_loss

    with open(
        config_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            config,
            f,
            indent=2,
        )

    save_accuracy_graph(
        history,
        graph_path,
    )

    print()
    print("=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)

    print(
        "Optimizer steps:",
        optimizer_step,
    )

    print(
        "Final checkpoint:",
        final_checkpoint_path,
    )

    print(
        "Best checkpoint:",
        best_checkpoint_path,
    )

    print(
        "Training log:",
        log_path,
    )

    print(
        "Configuration:",
        config_path,
    )

    print(
        "Accuracy graph:",
        graph_path,
    )

    print("=" * 60)


if __name__ == "__main__":
    main()
