import csv
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from datasets import load_dataset
from torchvision.models import ResNet18_Weights


# Official preprocessing associated with the requested
# torchvision ImageNet initialization.
RESNET_WEIGHTS = ResNet18_Weights.IMAGENET1K_V1
RESNET_TRANSFORM = RESNET_WEIGHTS.transforms()


class NIHDataset(Dataset):
    """
    NIH malaria dataset using explicitly supplied dataset indices.
    """

    def __init__(self, indices_file):
        self.indices_file = Path(indices_file)

        if not self.indices_file.exists():
            raise FileNotFoundError(self.indices_file)

        # Milestone-4 training subsets are JSON.
        # Frozen validation indices are NPY.
        if self.indices_file.suffix.lower() == ".json":
            with open(self.indices_file, "r", encoding="utf-8") as f:
                self.indices = [int(x) for x in json.load(f)]

        elif self.indices_file.suffix.lower() == ".npy":
            self.indices = [
                int(x)
                for x in np.load(self.indices_file)
            ]

        else:
            raise ValueError(
                f"Unsupported NIH index file: {self.indices_file}"
            )

        self.dataset = load_dataset(
            "dpdl-benchmark/malaria"
        )["train"]

        self.transform = RESNET_TRANSFORM

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        dataset_index = self.indices[idx]
        example = self.dataset[dataset_index]

        image = example["image"]

        if not isinstance(image, Image.Image):
            image = Image.fromarray(np.asarray(image))

        image = image.convert("RGB")
        image = self.transform(image)

        label = torch.tensor(
            float(example["label"]),
            dtype=torch.float32,
        )

        return image, label


class BBBC041Dataset(Dataset):
    """
    BBBC041 cell-crop dataset.

    subset_csv must contain crop_filename.

    Labels and metadata are recovered from the frozen processed
    training metadata, not inferred from filenames.
    """

    def __init__(
        self,
        subset_csv,
        metadata_csv="data/bbbc041_processed/train_metadata.csv",
        image_dir="data/bbbc041_processed/train",
    ):
        self.subset_csv = Path(subset_csv)
        self.metadata_csv = Path(metadata_csv)
        self.image_dir = Path(image_dir)

        if not self.subset_csv.exists():
            raise FileNotFoundError(self.subset_csv)

        if not self.metadata_csv.exists():
            raise FileNotFoundError(self.metadata_csv)

        if not self.image_dir.exists():
            raise FileNotFoundError(self.image_dir)

        with open(
            self.metadata_csv,
            newline="",
            encoding="utf-8",
        ) as f:
            metadata_rows = list(csv.DictReader(f))

        metadata = {
            row["crop_filename"]: row
            for row in metadata_rows
        }

        with open(
            self.subset_csv,
            newline="",
            encoding="utf-8",
        ) as f:
            subset_rows = list(csv.DictReader(f))

        self.records = []

        for row in subset_rows:
            filename = row["crop_filename"]

            if filename not in metadata:
                raise KeyError(
                    f"{filename} missing from training metadata."
                )

            meta = metadata[filename]

            self.records.append(
                {
                    "crop_filename": filename,
                    "label": int(meta["label"]),
                    "source_image": meta["source_image"],
                }
            )

        self.transform = RESNET_TRANSFORM

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        record = self.records[idx]

        image_path = self.image_dir / record["crop_filename"]

        if not image_path.exists():
            raise FileNotFoundError(image_path)

        with Image.open(image_path) as img:
            image = img.convert("RGB")
            image = self.transform(image)

        label = torch.tensor(
            float(record["label"]),
            dtype=torch.float32,
        )

        return image, label


def make_nih_train_dataset(fraction, seed):
    if fraction == 100:
        path = Path(
            "subsets_m4/nih/fraction_100.json"
        )
    else:
        path = Path(
            f"subsets_m4/nih/"
            f"fraction_{fraction}_seed_{seed}.json"
        )

    return NIHDataset(path)


def make_nih_val_dataset():
    return NIHDataset(
        "validation/nih_val_indices.npy"
    )


def make_bbbc041_train_dataset(fraction, seed):
    if fraction == 100:
        path = Path(
            "subsets_m4/bbbc041/fraction_100.csv"
        )
    else:
        path = Path(
            f"subsets_m4/bbbc041/"
            f"fraction_{fraction}_seed_{seed}.csv"
        )

    return BBBC041Dataset(path)


def make_bbbc041_val_dataset():
    return BBBC041Dataset(
        "validation/bbbc041_val_metadata.csv"
    )


def get_bbbc041_label_counts(dataset):
    labels = [
        int(record["label"])
        for record in dataset.records
    ]

    negative = labels.count(0)
    positive = labels.count(1)

    return negative, positive
