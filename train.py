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
from transformers import AutoImageProcessor, AutoModelForImageClassification

from dataset import (
    count_binary_targets,
    load_bbbc_splits,
    load_nih_splits,
)


MODEL_NAME = "ngohjuniormbah/plasmovision-malaria-ai"

BATCH_SIZE = 32
TOTAL_STEPS = 5000

# Fixed Milestone 4 optimization protocol.
LEARNING_RATE = 5e-5
WEIGHT_DECAY = 0.01

# Validate periodically by optimizer step rather than every epoch.
# This prevents tiny label fractions from requiring hundreds of
# full validation passes.
VALIDATE_EVERY = 250


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
        choices=[2, 10, 25, 100],
    )

    parser.add_argument(
        "--seed",
        required=True,
        type=int,
        choices=[42, 43],
    )

    parser.add_argument(
        "--steps",
        type=int,
        default=TOTAL_STEPS,
    )

    parser.add_argument(
        "--output-root",
        default="outputs/plasmovision",
    )

    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_model(seed):
    """
    Load the checkpoint's own weights, then replace its original
    2-output classifier with a fresh one-logit binary head.

    No backbone parameters are frozen.
    """

    # Make fresh-head initialization reproducible for this run.
    set_seed(seed)

    model = AutoModelForImageClassification.from_pretrained(
        MODEL_NAME
    )

    if not hasattr(model, "classifier"):
        raise RuntimeError(
            "Expected PlasmoVision model to expose model.classifier"
        )

    old_classifier = model.classifier

    if not hasattr(old_classifier, "in_features"):
        raise RuntimeError(
            "Could not determine classifier in_features"
        )

    in_features = old_classifier.in_features

    model.classifier = nn.Linear(in_features, 1)

    # Keep saved model configuration consistent with the new head.
    model.config.num_labels = 1

    # Explicitly ensure the entire model is trainable.
    for parameter in model.parameters():
        parameter.requires_grad = True

    frozen = [
        name
        for name, parameter in model.named_parameters()
        if not parameter.requires_grad
    ]

    if frozen:
        raise RuntimeError(
            f"Unexpected frozen parameters: {frozen[:10]}"
        )

    if model.classifier.out_features != 1:
        raise RuntimeError(
            "Fresh PlasmoVision classifier is not one-logit"
        )

    return model, in_features


def make_datasets(dataset_name, fraction, seed, processor):
    if dataset_name == "nih":
        return load_nih_splits(
            fraction,
            seed,
            processor,
        )

    if dataset_name == "bbbc041":
        return load_bbbc_splits(
            fraction,
            seed,
            processor,
        )

    raise ValueError(dataset_name)


def make_loaders(train_dataset, val_dataset, seed):
    generator = torch.Generator()
    generator.manual_seed(seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        generator=generator,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        drop_last=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    if len(train_loader) == 0:
        raise RuntimeError(
            "Training DataLoader contains zero full batches."
        )

    return train_loader, val_loader


def calculate_pos_weight(dataset_name, negative_count, positive_count):
    """
    BBBC041 protocol:
        pos_weight = negative examples / positive examples

    Positive means malaria-positive (target 1).
    """

    if dataset_name != "bbbc041":
        return None

    if positive_count == 0:
        raise RuntimeError(
            "Cannot calculate pos_weight: zero positive examples."
        )

    return negative_count / positive_count


def evaluate(model, loader, criterion, device):
    model.eval()

    total_loss = 0.0
    total_examples = 0

    targets_all = []
    predictions_all = []

    with torch.no_grad():
        for pixel_values, targets in loader:
            pixel_values = pixel_values.to(
                device,
                non_blocking=True,
            )

            targets = targets.to(
                device,
                dtype=torch.float32,
                non_blocking=True,
            )

            logits = model(
                pixel_values=pixel_values
            ).logits.squeeze(-1)

            loss = criterion(logits, targets)

            batch_size = targets.shape[0]

            total_loss += loss.item() * batch_size
            total_examples += batch_size

            predictions = (logits >= 0).long()

            targets_all.extend(
                targets.long().cpu().tolist()
            )

            predictions_all.extend(
                predictions.cpu().tolist()
            )

    if total_examples == 0:
        raise RuntimeError("Validation set is empty.")

    accuracy = np.mean(
        np.asarray(targets_all) ==
        np.asarray(predictions_all)
    )

    balanced_accuracy = balanced_accuracy_score(
        targets_all,
        predictions_all,
    )

    mcc = matthews_corrcoef(
        targets_all,
        predictions_all,
    )

    return {
        "loss": total_loss / total_examples,
        "accuracy": float(accuracy),
        "balanced_accuracy": float(balanced_accuracy),
        "mcc": float(mcc),
    }


def save_log(log_rows, path):
    if not log_rows:
        return

    fieldnames = list(log_rows[0].keys())

    with open(
        path,
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


def save_plot(log_rows, path):
    epochs = [row["epoch"] for row in log_rows]
    train_accuracy = [
        row["train_accuracy"]
        for row in log_rows
    ]
    val_accuracy = [
        row["val_accuracy"]
        for row in log_rows
    ]

    plt.figure(figsize=(8, 5))

    plt.plot(
        epochs,
        train_accuracy,
        label="Training accuracy",
    )

    plt.plot(
        epochs,
        val_accuracy,
        label="Validation accuracy",
    )

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training vs Validation Accuracy")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        path,
        dpi=150,
    )

    plt.close()


def main():
    args = parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. "
            "Milestone 4 PlasmoVision training should not "
            "silently fall back to CPU."
        )

    if args.steps <= 0:
        raise ValueError("--steps must be positive")

    set_seed(args.seed)

    device = torch.device("cuda")

    output_dir = (
        Path(args.output_root)
        / args.dataset
        / f"fraction_{args.fraction}"
        / f"seed_{args.seed}"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    processor = AutoImageProcessor.from_pretrained(
        MODEL_NAME
    )

    train_dataset, val_dataset = make_datasets(
        args.dataset,
        args.fraction,
        args.seed,
        processor,
    )

    negative_count, positive_count = count_binary_targets(
        train_dataset
    )

    pos_weight_value = calculate_pos_weight(
        args.dataset,
        negative_count,
        positive_count,
    )

    train_loader, val_loader = make_loaders(
        train_dataset,
        val_dataset,
        args.seed,
    )

    steps_per_epoch = len(train_loader)

    model, classifier_in_features = build_model(
        args.seed
    )

    model.to(device)

    if pos_weight_value is None:
        criterion = nn.BCEWithLogitsLoss()
    else:
        pos_weight_tensor = torch.tensor(
            [pos_weight_value],
            dtype=torch.float32,
            device=device,
        )

        criterion = nn.BCEWithLogitsLoss(
            pos_weight=pos_weight_tensor
        )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    config = {
        "model": "PlasmoVision",
        "checkpoint": MODEL_NAME,
        "dataset": args.dataset,
        "fraction_percent": args.fraction,
        "seed": args.seed,
        "batch_size": BATCH_SIZE,
        "optimizer_steps": args.steps,
        "optimizer": "AdamW",
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "scheduler": None,
        "loss": "BCEWithLogitsLoss",
        "pos_weight": pos_weight_value,
        "negative_training_examples": negative_count,
        "positive_training_examples": positive_count,
        "training_examples_after_validation_filter": len(
            train_dataset
        ),
        "validation_examples": len(val_dataset),
        "validation_seed": 2026,
        "validation_fraction": 0.10,
        "classifier_in_features": classifier_in_features,
        "classifier_out_features": 1,
        "full_backbone_trainable": True,
        "input_size": 224,
        "processor_class": processor.__class__.__name__,
        "image_mean": processor.image_mean,
        "image_std": processor.image_std,
        "do_resize": processor.do_resize,
        "do_rescale": processor.do_rescale,
        "do_normalize": processor.do_normalize,
        "rescale_factor": processor.rescale_factor,
        "validate_every_optimizer_steps": VALIDATE_EVERY,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
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

    print("=" * 70)
    print("PLASMOVISION MILESTONE 4")
    print("=" * 70)

    print(f"Dataset: {args.dataset}")
    print(f"Fraction: {args.fraction}%")
    print(f"Seed: {args.seed}")
    print(f"Training examples: {len(train_dataset)}")
    print(f"Validation examples: {len(val_dataset)}")
    print(f"Negative examples: {negative_count}")
    print(f"Positive examples: {positive_count}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Steps per epoch: {steps_per_epoch}")
    print(f"Required optimizer steps: {args.steps}")
    print(f"Classifier: {classifier_in_features} -> 1")
    print("Whole backbone trainable: YES")
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    if pos_weight_value is not None:
        print(
            "BBBC041 pos_weight: "
            f"{pos_weight_value:.6f}"
        )
    else:
        print("pos_weight: None")

    print("=" * 70)

    global_step = 0
    log_rows = []

    interval_loss_sum = 0.0
    interval_correct = 0
    interval_examples = 0

    model.train()

    while global_step < args.steps:
        for pixel_values, targets in train_loader:
            if global_step >= args.steps:
                break

            model.train()

            pixel_values = pixel_values.to(
                device,
                non_blocking=True,
            )

            targets = targets.to(
                device,
                dtype=torch.float32,
                non_blocking=True,
            )

            optimizer.zero_grad(set_to_none=True)

            logits = model(
                pixel_values=pixel_values
            ).logits.squeeze(-1)

            loss = criterion(
                logits,
                targets,
            )

            loss.backward()

            optimizer.step()

            global_step += 1

            batch_size = targets.shape[0]

            predictions = (logits.detach() >= 0).long()

            correct = (
                predictions ==
                targets.long()
            ).sum().item()

            interval_loss_sum += (
                loss.item() * batch_size
            )

            interval_correct += correct
            interval_examples += batch_size

            should_validate = (
                global_step % VALIDATE_EVERY == 0
                or global_step == args.steps
            )

            if should_validate:
                train_loss = (
                    interval_loss_sum /
                    interval_examples
                )

                train_accuracy = (
                    interval_correct /
                    interval_examples
                )

                val_metrics = evaluate(
                    model,
                    val_loader,
                    criterion,
                    device,
                )

                epoch = (
                    global_step /
                    steps_per_epoch
                )

                row = {
                    "optimizer_step": global_step,
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "train_accuracy": train_accuracy,
                    "val_loss": val_metrics["loss"],
                    "val_accuracy": val_metrics["accuracy"],
                    "val_balanced_accuracy":
                        val_metrics["balanced_accuracy"],
                    "val_mcc": val_metrics["mcc"],
                }

                log_rows.append(row)

                save_log(
                    log_rows,
                    output_dir / "training_log.csv",
                )

                save_plot(
                    log_rows,
                    output_dir / "accuracy_curve.png",
                )

                print(
                    f"step={global_step:5d}/{args.steps} "
                    f"epoch={epoch:8.3f} "
                    f"train_loss={train_loss:.5f} "
                    f"train_acc={train_accuracy:.4f} "
                    f"val_loss={val_metrics['loss']:.5f} "
                    f"val_acc={val_metrics['accuracy']:.4f} "
                    f"val_bal_acc="
                    f"{val_metrics['balanced_accuracy']:.4f} "
                    f"val_mcc={val_metrics['mcc']:.4f}"
                )

                interval_loss_sum = 0.0
                interval_correct = 0
                interval_examples = 0

                model.train()

    # Exactly args.steps optimizer updates have now occurred.
    if global_step != args.steps:
        raise RuntimeError(
            f"Expected {args.steps} optimizer steps, "
            f"but completed {global_step}."
        )

    checkpoint_dir = output_dir / "final_checkpoint"

    checkpoint_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    model.save_pretrained(
        checkpoint_dir,
        safe_serialization=True,
    )

    processor.save_pretrained(
        checkpoint_dir
    )

    final_summary = {
        "status": "complete",
        "optimizer_steps_completed": global_step,
        "final_epoch": global_step / steps_per_epoch,
        "final_train_loss": log_rows[-1]["train_loss"],
        "final_train_accuracy":
            log_rows[-1]["train_accuracy"],
        "final_validation_loss":
            log_rows[-1]["val_loss"],
        "final_validation_accuracy":
            log_rows[-1]["val_accuracy"],
        "final_validation_balanced_accuracy":
            log_rows[-1]["val_balanced_accuracy"],
        "final_validation_mcc":
            log_rows[-1]["val_mcc"],
    }

    with open(
        output_dir / "summary.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            final_summary,
            f,
            indent=2,
        )

    print()
    print("=" * 70)
    print("RUN COMPLETE")
    print(f"Optimizer steps completed: {global_step}")
    print(f"Output: {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()