import argparse
import csv
import json
import math
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import balanced_accuracy_score, matthews_corrcoef
from torch.utils.data import DataLoader

from dataset import (
    make_nih_train_dataset,
    make_nih_val_dataset,
    make_bbbc041_train_dataset,
    make_bbbc041_val_dataset,
    get_bbbc041_label_counts,
)
from models import build_resnet18


OFFICIAL_STEPS = 5000
OFFICIAL_BATCH_SIZE = 32


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        required=True,
        choices=["nih", "bbbc041"],
    )
    parser.add_argument(
        "--fraction",
        required=True,
        type=int,
        choices=[100, 25, 10, 2],
    )
    parser.add_argument(
        "--seed",
        required=True,
        type=int,
        choices=[42, 43],
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=5000,
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
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
    parser.add_argument(
        "--smoke-test",
        action="store_true",
    )

    return parser.parse_args()


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_datasets(name, fraction, seed):
    if name == "nih":
        train_dataset = make_nih_train_dataset(
            fraction,
            seed,
        )
        val_dataset = make_nih_val_dataset()

    else:
        train_dataset = make_bbbc041_train_dataset(
            fraction,
            seed,
        )
        val_dataset = make_bbbc041_val_dataset()

    return train_dataset, val_dataset


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    all_targets = []
    all_predictions = []

    for images, targets in loader:
        images = images.to(
            device,
            non_blocking=True,
        )

        targets = targets.to(
            device,
            non_blocking=True,
        )

        logits = model(images).squeeze(1)

        loss = criterion(logits, targets)

        batch_size = targets.size(0)

        total_loss += loss.item() * batch_size

        probabilities = torch.sigmoid(logits)
        predictions = (
            probabilities >= 0.5
        ).long()

        target_int = targets.long()

        total_correct += (
            predictions == target_int
        ).sum().item()

        total_examples += batch_size

        all_targets.extend(
            target_int.cpu().tolist()
        )
        all_predictions.extend(
            predictions.cpu().tolist()
        )

    loss = total_loss / total_examples
    accuracy = total_correct / total_examples

    balanced_accuracy = balanced_accuracy_score(
        all_targets,
        all_predictions,
    )

    mcc = matthews_corrcoef(
        all_targets,
        all_predictions,
    )

    return {
        "loss": float(loss),
        "accuracy": float(accuracy),
        "balanced_accuracy": float(
            balanced_accuracy
        ),
        "mcc": float(mcc),
    }


def save_plot(rows, output_path):
    if not rows:
        return

    epochs = [
        float(row["epoch"])
        for row in rows
    ]

    train_accuracy = [
        float(row["training_accuracy"])
        for row in rows
    ]

    val_accuracy = [
        float(row["validation_accuracy"])
        for row in rows
    ]

    plt.figure(figsize=(8, 5))

    plt.plot(
        epochs,
        train_accuracy,
        label="Training Accuracy",
    )

    plt.plot(
        epochs,
        val_accuracy,
        label="Validation Accuracy",
    )

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training and Validation Accuracy")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=150,
    )

    plt.close()


def main():
    args = parse_args()

    # -------------------------------------------------
    # Protocol guards
    # -------------------------------------------------

    if args.batch_size != OFFICIAL_BATCH_SIZE:
        raise ValueError(
            "Milestone 4 protocol requires "
            "batch size exactly 32."
        )

    if not args.smoke_test:
        if args.max_steps != OFFICIAL_STEPS:
            raise ValueError(
                "Official Milestone 4 runs require "
                "exactly 5000 optimizer steps."
            )

    if args.smoke_test and args.max_steps >= OFFICIAL_STEPS:
        raise ValueError(
            "Smoke test should use fewer than "
            "5000 optimizer steps."
        )

    if args.eval_every <= 0:
        raise ValueError(
            "--eval-every must be positive."
        )

    seed_everything(args.seed)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("=" * 70)
    print("MILESTONE 4 - RESNET-18")
    print("=" * 70)
    print("Dataset:", args.dataset)
    print("Fraction:", args.fraction)
    print("Seed:", args.seed)
    print("Device:", device)
    print("Batch size:", args.batch_size)
    print("Maximum optimizer steps:", args.max_steps)
    print("Smoke test:", args.smoke_test)

    # -------------------------------------------------
    # Data
    # -------------------------------------------------

    train_dataset, val_dataset = make_datasets(
        args.dataset,
        args.fraction,
        args.seed,
    )

    print("Training samples:", len(train_dataset))
    print("Validation samples:", len(val_dataset))

    generator = torch.Generator()
    generator.manual_seed(args.seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        generator=generator,
        drop_last=False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    batches_per_epoch = len(train_loader)

    if batches_per_epoch == 0:
        raise RuntimeError(
            "Training DataLoader contains zero batches."
        )

    # -------------------------------------------------
    # Model
    # -------------------------------------------------

    model, weights = build_resnet18()
    model = model.to(device)

    if model.fc.out_features != 1:
        raise RuntimeError(
            "ResNet classifier is not one-logit."
        )

    if not all(
        parameter.requires_grad
        for parameter in model.parameters()
    ):
        raise RuntimeError(
            "Not all ResNet parameters are trainable."
        )

    print("Weights:", weights)
    print("Classifier:", model.fc)
    print(
        "All parameters trainable:",
        all(
            p.requires_grad
            for p in model.parameters()
        ),
    )

    # -------------------------------------------------
    # Loss
    # -------------------------------------------------

    negative_count = None
    positive_count = None
    pos_weight_value = None

    if args.dataset == "bbbc041":
        (
            negative_count,
            positive_count,
        ) = get_bbbc041_label_counts(
            train_dataset
        )

        if positive_count <= 0:
            raise RuntimeError(
                "BBBC041 training subset contains "
                "zero positive examples."
            )

        pos_weight_value = (
            negative_count / positive_count
        )

        pos_weight = torch.tensor(
            [pos_weight_value],
            dtype=torch.float32,
            device=device,
        )

        criterion = nn.BCEWithLogitsLoss(
            pos_weight=pos_weight
        )

        print(
            "BBBC041 negative count:",
            negative_count,
        )
        print(
            "BBBC041 positive count:",
            positive_count,
        )
        print(
            "BBBC041 pos_weight:",
            pos_weight_value,
        )

    else:
        criterion = nn.BCEWithLogitsLoss()

    # -------------------------------------------------
    # Optimizer
    # -------------------------------------------------

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # -------------------------------------------------
    # Output directory
    # -------------------------------------------------

    run_name = (
        f"fraction_{args.fraction}"
        f"_seed_{args.seed}"
    )

    output_dir = (
        Path("outputs")
        / "resnet18"
        / args.dataset
        / run_name
    )

    if args.smoke_test:
        output_dir = (
            Path("outputs")
            / "smoke_tests"
            / "resnet18"
            / args.dataset
            / run_name
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    config = {
        "model": "resnet18",
        "initialization":
            "ResNet18_Weights.IMAGENET1K_V1",
        "classifier":
            "fresh Linear(512, 1)",
        "full_finetuning": True,
        "dataset": args.dataset,
        "fraction_percent": args.fraction,
        "seed": args.seed,
        "batch_size": args.batch_size,
        "max_optimizer_steps": args.max_steps,
        "official_optimizer_steps":
            OFFICIAL_STEPS,
        "input_size": [224, 224],
        "loss": "BCEWithLogitsLoss",
        "negative_count": negative_count,
        "positive_count": positive_count,
        "pos_weight": pos_weight_value,
        "learning_rate": args.lr,
        "weight_decay": args.weight_decay,
        "optimizer": "AdamW",
        "eval_every_optimizer_steps":
            args.eval_every,
        "training_samples":
            len(train_dataset),
        "validation_samples":
            len(val_dataset),
        "validation_used": True,
        "test_used_during_training": False,
        "smoke_test": args.smoke_test,
    }

    with open(
        output_dir / "config.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            config,
            f,
            indent=2,
        )

    # -------------------------------------------------
    # Training
    # -------------------------------------------------

    log_path = output_dir / "training_log.csv"

    fieldnames = [
        "optimizer_step",
        "epoch",
        "training_loss",
        "training_accuracy",
        "validation_loss",
        "validation_accuracy",
        "validation_balanced_accuracy",
        "validation_mcc",
    ]

    log_rows = []

    global_step = 0
    epoch = 0

    best_val_accuracy = -math.inf
    best_step = None

    running_loss = 0.0
    running_correct = 0
    running_examples = 0

    train_iterator = iter(train_loader)

    while global_step < args.max_steps:

        try:
            images, targets = next(
                train_iterator
            )

        except StopIteration:
            epoch += 1
            train_iterator = iter(train_loader)

            images, targets = next(
                train_iterator
            )

        # First pass through loader is epoch 1.
        current_epoch = epoch + 1

        model.train()

        images = images.to(
            device,
            non_blocking=True,
        )

        targets = targets.to(
            device,
            non_blocking=True,
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        logits = model(images).squeeze(1)

        loss = criterion(
            logits,
            targets,
        )

        loss.backward()
        optimizer.step()

        global_step += 1

        batch_size = targets.size(0)

        predictions = (
            torch.sigmoid(logits) >= 0.5
        ).long()

        correct = (
            predictions == targets.long()
        ).sum().item()

        running_loss += (
            loss.item() * batch_size
        )
        running_correct += correct
        running_examples += batch_size

        should_evaluate = (
            global_step == 1
            or global_step % args.eval_every == 0
            or global_step == args.max_steps
        )

        if should_evaluate:

            training_loss = (
                running_loss
                / running_examples
            )

            training_accuracy = (
                running_correct
                / running_examples
            )

            validation = evaluate(
                model,
                val_loader,
                criterion,
                device,
            )

            # Fractional epoch gives the actual
            # amount of dataset traversal at this step.
            epoch_value = (
                global_step
                / batches_per_epoch
            )

            row = {
                "optimizer_step":
                    global_step,
                "epoch":
                    epoch_value,
                "training_loss":
                    training_loss,
                "training_accuracy":
                    training_accuracy,
                "validation_loss":
                    validation["loss"],
                "validation_accuracy":
                    validation["accuracy"],
                "validation_balanced_accuracy":
                    validation[
                        "balanced_accuracy"
                    ],
                "validation_mcc":
                    validation["mcc"],
            }

            log_rows.append(row)

            with open(
                log_path,
                "w",
                newline="",
                encoding="utf-8",
            ) as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=fieldnames,
                )

                writer.writeheader()
                writer.writerows(log_rows)

            print(
                f"step={global_step}/{args.max_steps} "
                f"epoch={epoch_value:.3f} "
                f"train_loss={training_loss:.6f} "
                f"train_acc={training_accuracy:.4f} "
                f"val_loss={validation['loss']:.6f} "
                f"val_acc={validation['accuracy']:.4f} "
                f"val_bal_acc="
                f"{validation['balanced_accuracy']:.4f} "
                f"val_mcc={validation['mcc']:.4f}"
            )

            if (
                validation["accuracy"]
                > best_val_accuracy
            ):
                best_val_accuracy = (
                    validation["accuracy"]
                )
                best_step = global_step

                torch.save(
                    {
                        "model_state_dict":
                            model.state_dict(),
                        "optimizer_step":
                            global_step,
                        "validation_accuracy":
                            best_val_accuracy,
                        "config":
                            config,
                    },
                    output_dir
                    / "best_validation_checkpoint.pt",
                )

            running_loss = 0.0
            running_correct = 0
            running_examples = 0

    # -------------------------------------------------
    # Final checkpoint - MUST be step 5000 official
    # -------------------------------------------------

    torch.save(
        {
            "model_state_dict":
                model.state_dict(),
            "optimizer_state_dict":
                optimizer.state_dict(),
            "optimizer_step":
                global_step,
            "config":
                config,
        },
        output_dir / "final_checkpoint.pt",
    )

    save_plot(
        log_rows,
        output_dir / "accuracy_curve.png",
    )

    summary = {
        "completed_optimizer_steps":
            global_step,
        "required_optimizer_steps":
            args.max_steps,
        "best_validation_accuracy":
            best_val_accuracy,
        "best_validation_step":
            best_step,
        "final_logged_validation_accuracy":
            log_rows[-1][
                "validation_accuracy"
            ],
        "final_logged_validation_balanced_accuracy":
            log_rows[-1][
                "validation_balanced_accuracy"
            ],
        "final_logged_validation_mcc":
            log_rows[-1][
                "validation_mcc"
            ],
        "test_used": False,
        "completed": (
            global_step == args.max_steps
        ),
    }

    with open(
        output_dir / "summary.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            summary,
            f,
            indent=2,
        )

    print()
    print("=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(
        "Optimizer steps:",
        global_step,
    )
    print(
        "Best validation accuracy:",
        best_val_accuracy,
    )
    print(
        "Best validation step:",
        best_step,
    )
    print(
        "Output:",
        output_dir,
    )


if __name__ == "__main__":
    main()

