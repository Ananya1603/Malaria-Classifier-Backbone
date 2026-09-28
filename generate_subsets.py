import csv
import json
import os
from collections import Counter

from sklearn.model_selection import train_test_split

from data_utils import (
    get_nih_dataset,
    get_train_test_indices,
    get_stratified_subset,
)


FRACTIONS = [1.0, 0.25, 0.10, 0.02]
SEEDS = [42, 43]

BBBC_METADATA = "data/bbbc041_processed/train_metadata.csv"
OUTPUT_DIR = "subsets"


def fraction_name(fraction):
    return {
        1.0: "100",
        0.25: "25",
        0.10: "10",
        0.02: "2",
    }[fraction]


def save_json(values, path):
    if hasattr(values, "tolist"):
        values = values.tolist()

    with open(path, "w") as f:
        json.dump(values, f, indent=2)


def print_distribution(name, labels):
    counts = Counter(labels)
    print(f"\n{name}")
    print(f"Total: {len(labels)}")

    for label, count in sorted(counts.items(), key=lambda x: str(x[0])):
        percentage = 100 * count / len(labels)
        print(f"  label {label}: {count} ({percentage:.2f}%)")


def generate_nih_subsets():
    print("\n" + "=" * 60)
    print("NIH")
    print("=" * 60)

    dataset = get_nih_dataset()

    # Use ONLY the training portion of the existing train/test split.
    train_indices, _ = get_train_test_indices(
        dataset,
        test_size=0.2,
        seed=42,
    )

    os.makedirs(os.path.join(OUTPUT_DIR, "nih"), exist_ok=True)

    # 100% is identical regardless of random seed, so save it once.
    path = os.path.join(
        OUTPUT_DIR,
        "nih",
        "fraction_100.json",
    )
    save_json(train_indices, path)

    labels = [dataset[i]["label"] for i in train_indices]
    print_distribution("NIH | 100%", labels)
    print(f"Saved: {path}")

    # Smaller subsets are generated once for each required seed.
    for fraction in FRACTIONS[1:]:
        for seed in SEEDS:
            subset_indices = get_stratified_subset(
                dataset,
                train_indices,
                fraction,
                seed=seed,
            )

            path = os.path.join(
                OUTPUT_DIR,
                "nih",
                f"fraction_{fraction_name(fraction)}_seed_{seed}.json",
            )

            save_json(subset_indices, path)

            labels = [dataset[i]["label"] for i in subset_indices]

            print_distribution(
                f"NIH | {fraction_name(fraction)}% | seed={seed}",
                labels,
            )
            print(f"Saved: {path}")


def read_bbbc_metadata():
    with open(BBBC_METADATA, "r", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        raise ValueError("BBBC041 train_metadata.csv is empty.")

    if "crop_filename" not in rows[0]:
        raise ValueError(
            "Expected a 'crop_filename' column in train_metadata.csv."
        )

    if "label" not in rows[0]:
        raise ValueError(
            "Expected a 'label' column in train_metadata.csv."
        )

    return rows


def save_filenames(filenames, path):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["crop_filename"])

        for filename in filenames:
            writer.writerow([filename])


def generate_bbbc_subsets():
    print("\n" + "=" * 60)
    print("BBBC041")
    print("=" * 60)

    # train_metadata.csv already represents the BBBC041 train split.
    rows = read_bbbc_metadata()

    filenames = [row["crop_filename"] for row in rows]
    labels = [row["label"] for row in rows]

    os.makedirs(os.path.join(OUTPUT_DIR, "bbbc041"), exist_ok=True)

    # 100% subset: all filenames from the training metadata.
    path = os.path.join(
        OUTPUT_DIR,
        "bbbc041",
        "fraction_100.csv",
    )

    save_filenames(filenames, path)

    print_distribution("BBBC041 | 100%", labels)
    print(f"Saved: {path}")

    # Stratified smaller subsets.
    for fraction in FRACTIONS[1:]:
        for seed in SEEDS:
            subset_filenames, _ = train_test_split(
                filenames,
                train_size=fraction,
                random_state=seed,
                stratify=labels,
            )

            # Recover labels for verification.
            label_by_filename = {
                row["crop_filename"]: row["label"]
                for row in rows
            }

            subset_labels = [
                label_by_filename[filename]
                for filename in subset_filenames
            ]

            path = os.path.join(
                OUTPUT_DIR,
                "bbbc041",
                f"fraction_{fraction_name(fraction)}_seed_{seed}.csv",
            )

            save_filenames(subset_filenames, path)

            print_distribution(
                f"BBBC041 | {fraction_name(fraction)}% | seed={seed}",
                subset_labels,
            )
            print(f"Saved: {path}")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    generate_nih_subsets()
    generate_bbbc_subsets()

    print("\n" + "=" * 60)
    print("Milestone 3 subset generation complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
