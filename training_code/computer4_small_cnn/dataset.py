import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from datasets import load_dataset


ROOT = Path(__file__).resolve().parent

TRANSFORM = transforms.Compose([
    transforms.Resize((80, 80)),
    transforms.ToTensor(),
])


class NIHDataset(Dataset):
    def __init__(self, hf_dataset, indices):
        self.dataset = hf_dataset
        self.indices = [int(x) for x in indices]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        item = self.dataset[self.indices[idx]]
        image = item["image"].convert("RGB")
        image = TRANSFORM(image)
        label = torch.tensor(float(item["label"]), dtype=torch.float32)
        return image, label


class BBBC041Dataset(Dataset):
    def __init__(self, metadata, image_dir):
        self.metadata = metadata.reset_index(drop=True)
        self.image_dir = Path(image_dir)

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        image_path = self.image_dir / row["crop_filename"]

        with Image.open(image_path) as img:
            image = img.convert("RGB")
            image = TRANSFORM(image)

        label = torch.tensor(float(row["label"]), dtype=torch.float32)
        return image, label


def _read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_nih(fraction, seed):
    hf_dataset = load_dataset("dpdl-benchmark/malaria", split="train")

    val_indices = set(
        int(x) for x in _read_json(ROOT / "validation" / "nih_val_indices.json")
    )

    if int(fraction) == 100:
        manifest = ROOT / "subsets" / "nih" / "fraction_100.json"
    else:
        manifest = (
            ROOT / "subsets" / "nih" /
            f"fraction_{int(fraction)}_seed_{int(seed)}.json"
        )

    if not manifest.exists():
        raise FileNotFoundError(f"NIH subset manifest not found: {manifest}")

    subset_indices = [int(x) for x in _read_json(manifest)]

    # Preserve M3 manifests unchanged. Remove fixed validation examples only
    # when constructing the M4 training dataset.
    train_indices = [x for x in subset_indices if x not in val_indices]
    validation_indices = sorted(val_indices)

    train_dataset = NIHDataset(hf_dataset, train_indices)
    val_dataset = NIHDataset(hf_dataset, validation_indices)

    return train_dataset, val_dataset, {
        "manifest": str(manifest),
        "original_subset_size": len(subset_indices),
        "effective_train_size": len(train_indices),
        "validation_size": len(validation_indices),
    }


def load_bbbc041(fraction, seed):
    processed = ROOT / "data" / "bbbc041_processed"

    train_metadata = pd.read_csv(processed / "train_metadata.csv")
    val_metadata = pd.read_csv(
        ROOT / "validation" / "bbbc041_val_metadata.csv"
    )

    if int(fraction) == 100:
        manifest = ROOT / "subsets" / "bbbc041" / "fraction_100.csv"
    else:
        manifest = (
            ROOT / "subsets" / "bbbc041" /
            f"fraction_{int(fraction)}_seed_{int(seed)}.csv"
        )

    if not manifest.exists():
        raise FileNotFoundError(
            f"BBBC041 subset manifest not found: {manifest}"
        )

    subset = pd.read_csv(manifest)

    if "crop_filename" not in subset.columns:
        raise ValueError(
            f"{manifest} does not contain crop_filename."
        )

    subset_names = set(subset["crop_filename"].astype(str))
    selected = train_metadata[
        train_metadata["crop_filename"].astype(str).isin(subset_names)
    ].copy()

    if len(selected) != len(subset_names):
        raise RuntimeError(
            "Some BBBC041 subset filenames were not found in train_metadata.csv."
        )

    val_sources = set(val_metadata["source_image"].astype(str))

    # Source-image exclusion prevents crop leakage between train and validation.
    selected = selected[
        ~selected["source_image"].astype(str).isin(val_sources)
    ].copy()

    train_dataset = BBBC041Dataset(
        selected,
        processed / "train",
    )

    val_dataset = BBBC041Dataset(
        val_metadata,
        processed / "train",
    )

    label_counts = selected["label"].astype(int).value_counts().to_dict()
    n_negative = int(label_counts.get(0, 0))
    n_positive = int(label_counts.get(1, 0))

    if n_negative == 0 or n_positive == 0:
        raise RuntimeError(
            f"BBBC041 effective training subset has invalid class counts: "
            f"{label_counts}"
        )

    # BCE target 1 is the project's existing canonical label 1.
    # Therefore pos_weight follows the requested N(label=0)/N(label=1).
    pos_weight = n_negative / n_positive

    return train_dataset, val_dataset, {
        "manifest": str(manifest),
        "original_subset_size": len(subset_names),
        "effective_train_size": len(selected),
        "validation_size": len(val_metadata),
        "negative_label_0": n_negative,
        "positive_label_1": n_positive,
        "pos_weight": float(pos_weight),
    }


def load_experiment_datasets(dataset, fraction, seed):
    dataset = dataset.lower()

    if dataset == "nih":
        train_ds, val_ds, info = load_nih(fraction, seed)
        return train_ds, val_ds, info, None

    if dataset == "bbbc041":
        train_ds, val_ds, info = load_bbbc041(fraction, seed)
        return train_ds, val_ds, info, info["pos_weight"]

    raise ValueError(f"Unknown dataset: {dataset}")

