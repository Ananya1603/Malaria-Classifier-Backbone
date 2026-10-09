
import csv
import json
from pathlib import Path

source = Path.home() / "Desktop" / "Milestone 4 Results"
destination = Path("milestone4_results") / "all_runs_summary.csv"

models = ["DINOv2", "ResNet18", "PlasmoVision", "Small_CNN"]
rows = []

for model in models:
    folder = source / model

    for filename in sorted(folder.rglob("*.json")):
        expected_name = "config.json" if model == "DINOv2" else "summary.json"
        if filename.name != expected_name:
            continue

        with open(filename, encoding="utf-8") as f:
            data = json.load(f)

        relative = filename.parent.relative_to(folder)
        parts = relative.parts

        dataset = data.get("dataset")
        if not dataset:
            dataset = "bbbc041" if "bbbc041" in parts else "nih"

        fraction = data.get("fraction_percent")
        seed = data.get("training_seed", data.get("seed"))

        for part in parts:
            if fraction is None and part.startswith("fraction_"):
                fraction = part.split("_")[1]
            if seed is None and "seed_" in part:
                seed = part.split("seed_")[-1]

        steps = (
            data.get("completed_optimizer_steps")
            or data.get("optimizer_steps_completed")
            or data.get("optimizer_steps")
        )

        accuracy = data.get(
            "final_validation_accuracy",
            data.get("final_logged_validation_accuracy")
        )

        balanced_accuracy = data.get(
            "final_validation_balanced_accuracy",
            data.get("final_logged_validation_balanced_accuracy")
        )

        mcc = data.get(
            "final_validation_mcc",
            data.get("final_logged_validation_mcc")
        )

        if model == "DINOv2":
            log_path = filename.parent / "training_log.csv"

            with open(log_path, newline="", encoding="utf-8") as log_file:
                log_rows = list(csv.DictReader(log_file))

            final_row = max(
                log_rows,
                key=lambda row: int(row["optimizer_step"])
            )

            accuracy = float(final_row["validation_accuracy"])

        rows.append({
            "model": model,
            "dataset": dataset,
            "fraction_percent": fraction,
            "seed": seed,
            "optimizer_steps": steps,
            "validation_accuracy": accuracy,
            "validation_balanced_accuracy": balanced_accuracy,
            "validation_mcc": mcc,
            "source_file": str(filename.relative_to(source))
        })

fields = [
    "model",
    "dataset",
    "fraction_percent",
    "seed",
    "optimizer_steps",
    "validation_accuracy",
    "validation_balanced_accuracy",
    "validation_mcc",
    "source_file"
]

destination.parent.mkdir(parents=True, exist_ok=True)

with open(destination, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)

print("Saved:", destination.resolve())
print("Total runs:", len(rows))

for model in models:
    count = sum(row["model"] == model for row in rows)
    print(model, count)
