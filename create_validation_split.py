import csv
import json
import os
from collections import Counter, defaultdict

import numpy as np
from sklearn.model_selection import train_test_split

from data_utils import get_nih_dataset, get_train_test_indices


VAL_FRACTION = 0.10
VAL_SEED = 2026

VAL_DIR = os.path.join("data", "validation")
NIH_VAL_FILE = os.path.join(VAL_DIR, "nih_val_indices.json")
NIH_TRAIN_FILE = os.path.join(VAL_DIR, "nih_train_indices.json")

BBBC_TRAIN_METADATA = os.path.join(
    "data", "bbbc041_processed", "train_metadata.csv"
)
BBBC_VAL_FILE = os.path.join(VAL_DIR, "bbbc041_val_metadata.csv")
BBBC_TRAIN_FILE = os.path.join(VAL_DIR, "bbbc041_train_metadata.csv")
BBBC_VAL_SOURCE_FILE = os.path.join(
    VAL_DIR, "bbbc041_val_source_images.json"
)
BBBC_TRAIN_SOURCE_FILE = os.path.join(
    VAL_DIR, "bbbc041_train_source_images.json"
)


def save_json(values, path):
    if hasattr(values, "tolist"):
        values = values.tolist()

    with open(path, "w", encoding="utf-8") as f:
        json.dump(values, f, indent=2)


def print_counts(name, labels):
    counts = Counter(labels)
    total = len(labels)

    print(f"\n{name}")
    print(f"Total: {total}")

    for label in sorted(counts, key=str):
        count = counts[label]
        pct = 100.0 * count / total if total else 0.0
        print(f"  label {label}: {count} ({pct:.2f}%)")


def create_nih_validation():
    print("\n" + "=" * 60)
    print("NIH FIXED VALIDATION SPLIT")
    print("=" * 60)

    dataset = get_nih_dataset()

    # Reuse the already-frozen train/test split.
    frozen_train_indices, frozen_test_indices = get_train_test_indices(
        dataset,
        test_size=0.2,
        seed=42,
    )

    frozen_train_indices = np.asarray(frozen_train_indices)
    frozen_test_indices = np.asarray(frozen_test_indices)

    labels = np.asarray(
        [dataset[int(i)]["label"] for i in frozen_train_indices]
    )

    train_indices, val_indices = train_test_split(
        frozen_train_indices,
        test_size=VAL_FRACTION,
        random_state=VAL_SEED,
        stratify=labels,
    )

    train_indices = np.asarray(train_indices)
    val_indices = np.asarray(val_indices)

    # Safety checks.
    assert not (set(train_indices) & set(val_indices))
    assert not (set(val_indices) & set(frozen_test_indices))
    assert not (set(train_indices) & set(frozen_test_indices))

    reconstructed = set(train_indices) | set(val_indices)
    assert reconstructed == set(frozen_train_indices)

    save_json(train_indices, NIH_TRAIN_FILE)
    save_json(val_indices, NIH_VAL_FILE)

    train_labels = [dataset[int(i)]["label"] for i in train_indices]
    val_labels = [dataset[int(i)]["label"] for i in val_indices]

    print_counts("NIH post-validation training pool", train_labels)
    print_counts("NIH validation", val_labels)

    print(f"\nSaved: {NIH_TRAIN_FILE}")
    print(f"Saved: {NIH_VAL_FILE}")
    print("NIH test overlap with validation: 0")


def read_bbbc_metadata():
    with open(
        BBBC_TRAIN_METADATA,
        newline="",
        encoding="utf-8",
    ) as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = reader.fieldnames

    if not fieldnames:
        raise RuntimeError("BBBC041 metadata has no header.")

    return rows, fieldnames


def source_stratum(source_rows):
    """
    Build a source-image-level stratum from the labels of all crops
    belonging to that source image.

    This keeps validation splitting at the source-image level so cells
    from one microscope image can never be divided between train/val.
    """
    labels = [str(row["label"]) for row in source_rows]

    counts = Counter(labels)

    # Stable representation of the set of crop labels in this source image.
    return "|".join(sorted(counts.keys()))


def create_bbbc_validation():
    print("\n" + "=" * 60)
    print("BBBC041 FIXED SOURCE-IMAGE VALIDATION SPLIT")
    print("=" * 60)

    rows, fieldnames = read_bbbc_metadata()

    by_source = defaultdict(list)

    for row in rows:
        by_source[row["source_image"]].append(row)

    source_images = np.asarray(sorted(by_source.keys()))

    strata = np.asarray(
        [source_stratum(by_source[source]) for source in source_images]
    )

    print(f"Frozen BBBC041 training source images: {len(source_images)}")
    print(f"Frozen BBBC041 training crops: {len(rows)}")
    print(f"Source strata: {dict(Counter(strata))}")

    try:
        train_sources, val_sources = train_test_split(
            source_images,
            test_size=VAL_FRACTION,
            random_state=VAL_SEED,
            stratify=strata,
        )
    except ValueError as exc:
        raise RuntimeError(
            "Cannot make the requested stratified 10% BBBC041 "
            "source-image validation split safely. "
            "Do not fall back to crop-level splitting."
        ) from exc

    train_source_set = set(train_sources.tolist())
    val_source_set = set(val_sources.tolist())

    assert not (train_source_set & val_source_set)
    assert train_source_set | val_source_set == set(source_images.tolist())

    train_rows = [
        row for row in rows
        if row["source_image"] in train_source_set
    ]

    val_rows = [
        row for row in rows
        if row["source_image"] in val_source_set
    ]

    assert len(train_rows) + len(val_rows) == len(rows)

    with open(
        BBBC_TRAIN_FILE,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(train_rows)

    with open(
        BBBC_VAL_FILE,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(val_rows)

    save_json(sorted(train_source_set), BBBC_TRAIN_SOURCE_FILE)
    save_json(sorted(val_source_set), BBBC_VAL_SOURCE_FILE)

    print_counts(
        "BBBC041 post-validation training crops",
        [row["label"] for row in train_rows],
    )

    print_counts(
        "BBBC041 validation crops",
        [row["label"] for row in val_rows],
    )

    print(f"\nTraining source images: {len(train_source_set)}")
    print(f"Validation source images: {len(val_source_set)}")
    print("Train/validation source-image overlap: 0")

    print(f"\nSaved: {BBBC_TRAIN_FILE}")
    print(f"Saved: {BBBC_VAL_FILE}")
    print(f"Saved: {BBBC_TRAIN_SOURCE_FILE}")
    print(f"Saved: {BBBC_VAL_SOURCE_FILE}")


def main():
    os.makedirs(VAL_DIR, exist_ok=True)

    print("Milestone 4 fixed validation protocol")
    print(f"Validation fraction: {VAL_FRACTION}")
    print(f"Validation seed: {VAL_SEED}")
    print("FINAL TEST SETS WILL NOT BE MODIFIED.")

    create_nih_validation()
    create_bbbc_validation()

    print("\n" + "=" * 60)
    print("FIXED VALIDATION SPLITS CREATED SUCCESSFULLY")
    print("=" * 60)


if __name__ == "__main__":
    main()