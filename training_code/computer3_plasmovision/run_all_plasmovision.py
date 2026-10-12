import subprocess
import sys
from pathlib import Path

DATASETS = ["nih", "bbbc041"]
FRACTIONS = [2, 10, 25, 100]
SEEDS = [42, 43]

completed = []
failed = []
skipped = []

for dataset in DATASETS:
    for fraction in FRACTIONS:
        for seed in SEEDS:

            output_dir = (
                Path("outputs")
                / "plasmovision"
                / dataset
                / f"fraction_{fraction}"
                / f"seed_{seed}"
            )

            summary_file = output_dir / "summary.json"

            # Do not repeat an already completed official run.
            if summary_file.exists():
                print()
                print("=" * 70)
                print(
                    f"SKIPPING completed run: "
                    f"{dataset} fraction={fraction}% seed={seed}"
                )
                print("=" * 70)

                skipped.append(
                    (dataset, fraction, seed)
                )
                continue

            print()
            print("=" * 70)
            print(
                f"STARTING: {dataset} "
                f"fraction={fraction}% seed={seed}"
            )
            print("=" * 70)

            command = [
                sys.executable,
                "train.py",
                "--dataset",
                dataset,
                "--fraction",
                str(fraction),
                "--seed",
                str(seed),
            ]

            result = subprocess.run(command)

            if result.returncode == 0 and summary_file.exists():
                print(
                    f"SUCCESS: {dataset} "
                    f"fraction={fraction}% seed={seed}"
                )

                completed.append(
                    (dataset, fraction, seed)
                )

            else:
                print(
                    f"FAILED: {dataset} "
                    f"fraction={fraction}% seed={seed}"
                )

                failed.append(
                    (dataset, fraction, seed)
                )

                # Stop here rather than silently continuing after a failure.
                break

        if failed:
            break

    if failed:
        break


print()
print("=" * 70)
print("PLASMOVISION LAUNCHER FINISHED")
print("=" * 70)

print(f"Newly completed: {len(completed)}")
print(f"Already completed / skipped: {len(skipped)}")
print(f"Failed: {len(failed)}")

if completed:
    print("\nNEWLY COMPLETED:")
    for dataset, fraction, seed in completed:
        print(
            f"  {dataset} fraction={fraction}% seed={seed}"
        )

if skipped:
    print("\nSKIPPED:")
    for dataset, fraction, seed in skipped:
        print(
            f"  {dataset} fraction={fraction}% seed={seed}"
        )

if failed:
    print("\nFAILED:")
    for dataset, fraction, seed in failed:
        print(
            f"  {dataset} fraction={fraction}% seed={seed}"
        )

    sys.exit(1)

print("\nALL REQUESTED RUNS ARE COMPLETE.")