import json
from pathlib import Path

import numpy as np
import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split


SEED = 2026
VAL_FRACTION = 0.10

OUT = Path("validation")
OUT.mkdir(exist_ok=True)


def create_nih_validation():
    print("\n" + "=" * 70)
    print("NIH VALIDATION")
    print("=" * 70)

    full_train_path = Path("subsets/nih/fraction_100.json")
    test_indices_path = Path("test_indices.npy")

    if not full_train_path.exists():
        raise FileNotFoundError(full_train_path)
    if not test_indices_path.exists():
        raise FileNotFoundError(test_indices_path)

    with open(full_train_path, "r") as f:
        train_indices = np.asarray(json.load(f), dtype=np.int64)

    test_indices = np.asarray(np.load(test_indices_path), dtype=np.int64)

    print("Frozen train indices:", len(train_indices))
    print("Frozen test indices :", len(test_indices))

    overlap = set(train_indices.tolist()) & set(test_indices.tolist())
    print("Existing train/test overlap:", len(overlap))

    if overlap:
        raise RuntimeError("NIH frozen train/test split contains overlap.")

    # Load labels only so validation can be stratified.
    ds = load_dataset("dpdl-benchmark/malaria", split="train")
    labels = np.asarray(ds["label"], dtype=np.int64)

    train_labels = labels[train_indices]

    remaining_indices, val_indices = train_test_split(
        train_indices,
        test_size=VAL_FRACTION,
        random_state=SEED,
        stratify=train_labels,
    )

    val_indices = np.asarray(sorted(val_indices.tolist()), dtype=np.int64)
    remaining_indices = np.asarray(
        sorted(remaining_indices.tolist()), dtype=np.int64
    )

    val_labels = labels[val_indices]

    print("New training pool:", len(remaining_indices))
    print("Validation size  :", len(val_indices))
    print(
        "Validation labels:",
        dict(zip(*np.unique(val_labels, return_counts=True))),
    )

    val_test_overlap = set(val_indices.tolist()) & set(test_indices.tolist())
    train_val_overlap = set(remaining_indices.tolist()) & set(val_indices.tolist())

    print("Validation/test overlap:", len(val_test_overlap))
    print("Train/validation overlap:", len(train_val_overlap))

    if val_test_overlap:
        raise RuntimeError("NIH validation leaks into frozen test set.")

    if train_val_overlap:
        raise RuntimeError("NIH validation leaks into remaining training pool.")

    if len(remaining_indices) + len(val_indices) != len(train_indices):
        raise RuntimeError("NIH split size mismatch.")

    with open(OUT / "nih_val_indices.json", "w") as f:
        json.dump(val_indices.tolist(), f, indent=2)

    with open(OUT / "nih_train_pool_indices.json", "w") as f:
        json.dump(remaining_indices.tolist(), f, indent=2)

    print("Saved NIH validation artifacts.")


def create_bbbc_validation():
    print("\n" + "=" * 70)
    print("BBBC041 VALIDATION")
    print("=" * 70)

    train_meta_path = Path(
        "data/bbbc041_processed/train_metadata.csv"
    )
    test_meta_path = Path(
        "data/bbbc041_processed/test_metadata.csv"
    )

    if not train_meta_path.exists():
        raise FileNotFoundError(train_meta_path)
    if not test_meta_path.exists():
        raise FileNotFoundError(test_meta_path)

    train_meta = pd.read_csv(train_meta_path)
    test_meta = pd.read_csv(test_meta_path)

    train_sources = np.asarray(
        sorted(train_meta["source_image"].unique())
    )
    test_sources = set(test_meta["source_image"].unique())

    print("Frozen train source images:", len(train_sources))
    print("Frozen test source images :", len(test_sources))

    existing_overlap = set(train_sources) & test_sources
    print("Existing train/test source overlap:", len(existing_overlap))

    if existing_overlap:
        raise RuntimeError(
            "BBBC041 frozen train/test split has source-image leakage."
        )

    # Source-level stratification:
    # A source is positive (0) if it contains at least one parasitized crop.
    # Otherwise it is uninfected-only (1).
    source_labels = (
        train_meta.groupby("source_image")["label"]
        .min()
        .reindex(train_sources)
        .to_numpy(dtype=np.int64)
    )

    remaining_sources, val_sources = train_test_split(
        train_sources,
        test_size=VAL_FRACTION,
        random_state=SEED,
        stratify=source_labels,
    )

    remaining_sources = np.asarray(sorted(remaining_sources))
    val_sources = np.asarray(sorted(val_sources))

    val_meta = train_meta[
        train_meta["source_image"].isin(val_sources)
    ].copy()

    remaining_meta = train_meta[
        train_meta["source_image"].isin(remaining_sources)
    ].copy()

    print("New training source pool:", len(remaining_sources))
    print("Validation source images:", len(val_sources))
    print("New training crops:", len(remaining_meta))
    print("Validation crops  :", len(val_meta))
    print(
        "Validation crop labels:",
        val_meta["label"].value_counts().sort_index().to_dict(),
    )

    val_test_overlap = set(val_sources) & test_sources
    train_val_overlap = set(remaining_sources) & set(val_sources)

    print("Validation/test source overlap:", len(val_test_overlap))
    print("Train/validation source overlap:", len(train_val_overlap))

    if val_test_overlap:
        raise RuntimeError(
            "BBBC041 validation source images leak into test."
        )

    if train_val_overlap:
        raise RuntimeError(
            "BBBC041 validation source images leak into training."
        )

    if len(remaining_sources) + len(val_sources) != len(train_sources):
        raise RuntimeError("BBBC041 source split size mismatch.")

    if len(remaining_meta) + len(val_meta) != len(train_meta):
        raise RuntimeError("BBBC041 crop split size mismatch.")

    pd.DataFrame(
        {"source_image": val_sources}
    ).to_csv(
        OUT / "bbbc041_val_source_images.csv",
        index=False,
    )

    pd.DataFrame(
        {"source_image": remaining_sources}
    ).to_csv(
        OUT / "bbbc041_train_pool_source_images.csv",
        index=False,
    )

    val_meta.to_csv(
        OUT / "bbbc041_val_metadata.csv",
        index=False,
    )

    print("Saved BBBC041 validation artifacts.")


if __name__ == "__main__":
    create_nih_validation()
    create_bbbc_validation()

    print("\n" + "=" * 70)
    print("VALIDATION SPLITS CREATED SUCCESSFULLY")
    print("=" * 70)
    print("Seed:", SEED)
    print("Validation fraction:", VAL_FRACTION)
    print("Frozen test sets were NOT modified.")
