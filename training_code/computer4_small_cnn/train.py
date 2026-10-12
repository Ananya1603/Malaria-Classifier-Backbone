import argparse
import csv
import json
import math
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score, matthews_corrcoef
from torch.utils.data import DataLoader

from dataset import load_experiment_datasets
from models import build_small_cnn


BATCH_SIZE = 32
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()

    total_loss = 0.0
    total_examples = 0
    all_targets = []
    all_predictions = []

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        logits = model(images)
        loss = criterion(logits, targets)

        batch_size = targets.numel()
        total_loss += loss.item() * batch_size
        total_examples += batch_size

        predictions = (logits >= 0).long()

        all_targets.extend(
            targets.detach().cpu().long().tolist()
        )
        all_predictions.extend(
            predictions.detach().cpu().tolist()
        )

    avg_loss = total_loss / total_examples
    targets_np = np.asarray(all_targets)
    predictions_np = np.asarray(all_predictions)

    accuracy = float((targets_np == predictions_np).mean())
    balanced_accuracy = float(
        balanced_accuracy_score(targets_np, predictions_np)
    )
    mcc = float(
        matthews_corrcoef(targets_np, predictions_np)
    )

    return avg_loss, accuracy, balanced_accuracy, mcc


def save_curve(rows, output_path):
    epochs = [row["epoch"] for row in rows]
    train_acc = [row["train_accuracy"] for row in rows]
    val_acc = [row["validation_accuracy"] for row in rows]

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_acc, label="Training Accuracy")
    plt.plot(epochs, val_acc, label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training vs Validation Accuracy")
    plt.ylim(0.0, 1.0)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def main():
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
        "--max-steps",
        type=int,
        default=5000,
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
    )

    args = parser.parse_args()

    if not args.smoke_test and args.max_steps != 5000:
        raise ValueError(
            "Official Small CNN experiments require exactly "
            "5000 optimizer steps."
        )

    set_seed(args.seed)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("=" * 70)
    print("MILESTONE 4 - SMALL CNN")
    print("=" * 70)
    print("Dataset:", args.dataset)
    print("Fraction:", args.fraction)
    print("Seed:", args.seed)
    print("Device:", device)
    print("Batch size:", BATCH_SIZE)
    print("Maximum optimizer steps:", args.max_steps)
    print("Smoke test:", args.smoke_test)

    train_dataset, val_dataset, dataset_info, pos_weight = (
        load_experiment_datasets(
            args.dataset,
            args.fraction,
            args.seed,
        )
    )

    if len(train_dataset) == 0:
        raise RuntimeError("Training dataset is empty.")

    print("Training samples:", len(train_dataset))
    print("Validation samples:", len(val_dataset))
    print("Dataset info:", dataset_info)

    generator = torch.Generator()
    generator.manual_seed(args.seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        generator=generator,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    # Seed was set before this call, so every experiment begins
    # from a fresh seed-specific random initialization.
    model = build_small_cnn().to(device)

    if args.dataset == "bbbc041":
        pos_weight_tensor = torch.tensor(
            [float(pos_weight)],
            dtype=torch.float32,
            device=device,
        )
        criterion = torch.nn.BCEWithLogitsLoss(
            pos_weight=pos_weight_tensor
        )
        print("BBBC041 pos_weight:", pos_weight)
    else:
        criterion = torch.nn.BCEWithLogitsLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    if args.smoke_test:
        output_dir = (
            Path("outputs")
            / "smoke_test"
            / "small_cnn"
            / args.dataset
            / f"fraction_{args.fraction}"
            / f"seed_{args.seed}"
        )
    else:
        output_dir = (
            Path("outputs")
            / "small_cnn"
            / args.dataset
            / f"fraction_{args.fraction}"
            / f"seed_{args.seed}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "model": "small_cnn",
        "initialization": "from_scratch",
        "dataset": args.dataset,
        "fraction_percent": args.fraction,
        "seed": args.seed,
        "input_size": [3, 80, 80],
        "batch_size": BATCH_SIZE,
        "max_optimizer_steps": args.max_steps,
        "optimizer": "Adam",
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "loss": "BCEWithLogitsLoss",
        "pos_weight": (
            float(pos_weight)
            if pos_weight is not None
            else None
        ),
        "augmentation": "none",
        "preprocessing": "RGB resize 80x80 + ToTensor [0,1]",
        "validation_protocol": (
            "Fixed 10% validation split from frozen M2 training "
            "split; validation seed 2026; validation examples "
            "excluded from M3 subset at M4 training time; frozen "
            "test set untouched."
        ),
        "dataset_info": dataset_info,
        "device": str(device),
        "smoke_test": bool(args.smoke_test),
    }

    with open(
        output_dir / "config.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(config, f, indent=2)

    rows = []
    optimizer_step = 0
    epoch = 0

    while optimizer_step < args.max_steps:
        epoch += 1
        model.train()

        epoch_loss_sum = 0.0
        epoch_examples = 0
        epoch_correct = 0

        for images, targets in train_loader:
            if optimizer_step >= args.max_steps:
                break

            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            logits = model(images)
            loss = criterion(logits, targets)

            loss.backward()
            optimizer.step()

            optimizer_step += 1

            batch_n = targets.numel()
            epoch_loss_sum += loss.item() * batch_n
            epoch_examples += batch_n

            predictions = (logits.detach() >= 0).long()
            epoch_correct += (
                predictions == targets.long()
            ).sum().item()

        train_loss = epoch_loss_sum / epoch_examples
        train_accuracy = epoch_correct / epoch_examples

        val_loss, val_accuracy, val_balanced_accuracy, val_mcc = (
            evaluate(
                model,
                val_loader,
                criterion,
                device,
            )
        )

        row = {
            "optimizer_step": optimizer_step,
            "epoch": epoch,
            "train_loss": float(train_loss),
            "train_accuracy": float(train_accuracy),
            "validation_loss": float(val_loss),
            "validation_accuracy": float(val_accuracy),
            "validation_balanced_accuracy": float(
                val_balanced_accuracy
            ),
            "validation_mcc": float(val_mcc),
        }

        rows.append(row)

        print(
            f"epoch={epoch:04d} "
            f"step={optimizer_step:04d}/{args.max_steps} "
            f"train_loss={train_loss:.6f} "
            f"train_acc={train_accuracy:.4f} "
            f"val_loss={val_loss:.6f} "
            f"val_acc={val_accuracy:.4f} "
            f"val_bal_acc={val_balanced_accuracy:.4f} "
            f"val_mcc={val_mcc:.4f}",
            flush=True,
        )

    if optimizer_step != args.max_steps:
        raise RuntimeError(
            f"Expected {args.max_steps} optimizer steps, "
            f"but completed {optimizer_step}."
        )

    log_path = output_dir / "training_log.csv"

    with open(
        log_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(rows)

    save_curve(
        rows,
        output_dir / "accuracy_curve.png",
    )

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": config,
            "final_optimizer_step": optimizer_step,
            "final_epoch": epoch,
        },
        output_dir / "final_model.pt",
    )

    summary = {
        "status": "complete",
        "dataset": args.dataset,
        "fraction_percent": args.fraction,
        "seed": args.seed,
        "optimizer_steps": optimizer_step,
        "epochs": epoch,
        "effective_train_size": len(train_dataset),
        "validation_size": len(val_dataset),
        "final_train_loss": rows[-1]["train_loss"],
        "final_train_accuracy": rows[-1]["train_accuracy"],
        "final_validation_loss": rows[-1]["validation_loss"],
        "final_validation_accuracy": rows[-1]["validation_accuracy"],
        "final_validation_balanced_accuracy": rows[-1][
            "validation_balanced_accuracy"
        ],
        "final_validation_mcc": rows[-1]["validation_mcc"],
        "pos_weight": (
            float(pos_weight)
            if pos_weight is not None
            else None
        ),
    }

    with open(
        output_dir / "summary.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(summary, f, indent=2)

    print("=" * 70)
    print("RUN COMPLETE")
    print("Optimizer steps:", optimizer_step)
    print("Output:", output_dir)
    print("=" * 70)


if __name__ == "__main__":
    main()
