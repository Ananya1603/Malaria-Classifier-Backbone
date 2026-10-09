import csv
import json
import subprocess
import sys
from pathlib import Path


DATASETS = ["nih", "bbbc041"]
FRACTIONS = [2, 10, 25, 100]
SEEDS = [42, 43]
MAX_STEPS = 5000
MAX_ATTEMPTS = 2

ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = ROOT / "outputs" / "small_cnn"


def output_dir(dataset, fraction, seed):
    return (
        OUTPUT_ROOT
        / dataset
        / f"fraction_{fraction}"
        / f"seed_{seed}"
    )


def is_complete(dataset, fraction, seed):
    out = output_dir(dataset, fraction, seed)

    required = [
        out / "config.json",
        out / "training_log.csv",
        out / "accuracy_curve.png",
        out / "final_model.pt",
        out / "summary.json",
    ]

    if not all(path.exists() for path in required):
        return False

    try:
        with open(out / "summary.json", "r", encoding="utf-8") as f:
            summary = json.load(f)

        with open(out / "config.json", "r", encoding="utf-8") as f:
            config = json.load(f)

        return (
            summary.get("status") == "complete"
            and summary.get("optimizer_steps") == MAX_STEPS
            and config.get("max_optimizer_steps") == MAX_STEPS
            and config.get("smoke_test") is False
        )
    except Exception:
        return False


def run_cell(dataset, fraction, seed):
    command = [
        sys.executable,
        str(ROOT / "train.py"),
        "--dataset", dataset,
        "--fraction", str(fraction),
        "--seed", str(seed),
        "--max-steps", str(MAX_STEPS),
        "--num-workers", "0",
    ]

    for attempt in range(1, MAX_ATTEMPTS + 1):
        print("\n" + "#" * 78)
        print(
            f"RUNNING: {dataset} | fraction={fraction}% | "
            f"seed={seed} | attempt={attempt}/{MAX_ATTEMPTS}"
        )
        print("#" * 78, flush=True)

        result = subprocess.run(command, cwd=ROOT)

        if result.returncode == 0 and is_complete(
            dataset, fraction, seed
        ):
            print("VERIFIED COMPLETE")
            return True

        print(
            f"Attempt {attempt} failed or produced incomplete outputs.",
            flush=True,
        )

    return False


def collect_summary():
    rows = []

    for dataset in DATASETS:
        for fraction in FRACTIONS:
            for seed in SEEDS:
                out = output_dir(dataset, fraction, seed)
                summary_path = out / "summary.json"

                if is_complete(dataset, fraction, seed):
                    with open(
                        summary_path,
                        "r",
                        encoding="utf-8",
                    ) as f:
                        summary = json.load(f)

                    rows.append({
                        "dataset": dataset,
                        "fraction": fraction,
                        "seed": seed,
                        "status": "complete",
                        "optimizer_steps": summary[
                            "optimizer_steps"
                        ],
                        "effective_train_size": summary[
                            "effective_train_size"
                        ],
                        "validation_size": summary[
                            "validation_size"
                        ],
                        "validation_accuracy": summary[
                            "final_validation_accuracy"
                        ],
                        "validation_balanced_accuracy": summary[
                            "final_validation_balanced_accuracy"
                        ],
                        "validation_mcc": summary[
                            "final_validation_mcc"
                        ],
                        "pos_weight": summary["pos_weight"],
                    })
                else:
                    rows.append({
                        "dataset": dataset,
                        "fraction": fraction,
                        "seed": seed,
                        "status": "FAILED_OR_INCOMPLETE",
                        "optimizer_steps": "",
                        "effective_train_size": "",
                        "validation_size": "",
                        "validation_accuracy": "",
                        "validation_balanced_accuracy": "",
                        "validation_mcc": "",
                        "pos_weight": "",
                    })

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    summary_csv = OUTPUT_ROOT / "all_runs_summary.csv"

    with open(
        summary_csv,
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

    return rows


def main():
    print("=" * 78)
    print("SMALL CNN - COMPLETE MILESTONE 4 RUNNER")
    print("=" * 78)
    print("Datasets :", DATASETS)
    print("Fractions:", FRACTIONS)
    print("Seeds    :", SEEDS)
    print("Runs     :", 16)
    print("Steps/run:", MAX_STEPS)
    print("=" * 78)

    failures = []

    for dataset in DATASETS:
        for fraction in FRACTIONS:
            for seed in SEEDS:
                if is_complete(dataset, fraction, seed):
                    print(
                        f"SKIPPING VERIFIED COMPLETE: "
                        f"{dataset} fraction={fraction}% seed={seed}"
                    )
                    continue

                success = run_cell(
                    dataset,
                    fraction,
                    seed,
                )

                if not success:
                    failures.append(
                        (dataset, fraction, seed)
                    )

    rows = collect_summary()

    complete_count = sum(
        row["status"] == "complete"
        for row in rows
    )

    print("\n" + "=" * 78)
    print("FINAL SMALL CNN STATUS")
    print("=" * 78)
    print(f"Verified complete: {complete_count}/16")

    if failures:
        print("Failed after retry:")
        for item in failures:
            print(" ", item)

    if complete_count == 16:
        print("ALL 16 SMALL CNN EXPERIMENTS COMPLETE.")
        sys.exit(0)

    print("ONE OR MORE EXPERIMENTS ARE INCOMPLETE.")
    sys.exit(1)


if __name__ == "__main__":
    main()
