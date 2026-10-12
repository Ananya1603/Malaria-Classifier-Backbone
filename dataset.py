import csv
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from data_utils import get_nih_dataset


MODEL_IMAGE_SIZES = {
    "dinov2": 518,
    "plasmovision": 224,
    "resnet18": 224,
    "small_cnn": 80,
}

# Verified from:
# Sadou/malaria-detector-dinov2-tanzania
DINO_MEAN = [0.485, 0.456, 0.406]
DINO_STD = [0.229, 0.224, 0.225]


def get_transform(model_name):
    if model_name not in MODEL_IMAGE_SIZES:
        raise ValueError(
            f"Unknown model: {model_name}"
        )

    size = MODEL_IMAGE_SIZES[model_name]

    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=DINO_MEAN,
            std=DINO_STD,
        ),
    ])


def get_subset_path(dataset_name, fraction, seed):
    fraction = int(fraction)

    if dataset_name == "nih":
        base = Path("subsets/nih")

        if fraction == 100:
            return base / "fraction_100.json"

        return (
            base
            / f"fraction_{fraction}_seed_{seed}.json"
        )

    if dataset_name == "bbbc041":
        base = Path("subsets/bbbc041")

        if fraction == 100:
            return base / "fraction_100.csv"

        return (
            base
            / f"fraction_{fraction}_seed_{seed}.csv"
        )

    raise ValueError(
        f"Unknown dataset: {dataset_name}"
    )


class NIHDataset(Dataset):
    def __init__(
        self,
        model_name,
        fraction=None,
        seed=None,
        validation=False,
    ):
        self.dataset = get_nih_dataset()
        self.transform = get_transform(model_name)
        self.validation = validation

        if validation:
            path = Path("val_indices.npy")

            if not path.exists():
                raise FileNotFoundError(
                    "val_indices.npy not found."
                )

            self.indices = (
                np.load(path)
                .astype(int)
                .tolist()
            )

        else:
            subset_path = get_subset_path(
                "nih",
                fraction,
                seed,
            )

            if not subset_path.exists():
                raise FileNotFoundError(
                    f"NIH subset not found: "
                    f"{subset_path}"
                )

            with open(
                subset_path,
                "r",
                encoding="utf-8",
            ) as f:
                self.indices = [
                    int(i)
                    for i in json.load(f)
                ]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        dataset_idx = self.indices[idx]
        sample = self.dataset[dataset_idx]

        image = sample["image"].convert("RGB")
        label = float(sample["label"])

        image = self.transform(image)

        return (
            image,
            torch.tensor(
                label,
                dtype=torch.float32,
            ),
        )


class BBBC041Dataset(Dataset):
    def __init__(
        self,
        model_name,
        fraction=None,
        seed=None,
        validation=False,
    ):
        self.transform = get_transform(model_name)

        # Validation crops physically remain in the
        # original processed/train directory.
        self.crop_dir = Path(
            "data/bbbc041_processed/train"
        )

        if validation:
            metadata_path = Path(
                "data/bbbc041_processed/"
                "val_metadata.csv"
            )

            if not metadata_path.exists():
                raise FileNotFoundError(
                    f"Missing {metadata_path}"
                )

            self.filenames = []
            self.label_map = {}

            with open(
                metadata_path,
                "r",
                newline="",
                encoding="utf-8",
            ) as f:
                reader = csv.DictReader(f)

                for row in reader:
                    filename = row["crop_filename"]

                    self.filenames.append(
                        filename
                    )

                    self.label_map[filename] = int(
                        row["label"]
                    )

        else:
            metadata_path = Path(
                "data/bbbc041_processed/"
                "train_pool_metadata.csv"
            )

            subset_path = get_subset_path(
                "bbbc041",
                fraction,
                seed,
            )

            if not metadata_path.exists():
                raise FileNotFoundError(
                    f"Missing {metadata_path}"
                )

            if not subset_path.exists():
                raise FileNotFoundError(
                    f"Missing {subset_path}"
                )

            self.label_map = {}

            with open(
                metadata_path,
                "r",
                newline="",
                encoding="utf-8",
            ) as f:
                reader = csv.DictReader(f)

                for row in reader:
                    self.label_map[
                        row["crop_filename"]
                    ] = int(row["label"])

            self.filenames = []

            with open(
                subset_path,
                "r",
                newline="",
                encoding="utf-8",
            ) as f:
                reader = csv.DictReader(f)

                for row in reader:
                    self.filenames.append(
                        row["crop_filename"]
                    )

            missing = [
                name
                for name in self.filenames
                if name not in self.label_map
            ]

            if missing:
                raise RuntimeError(
                    f"{len(missing)} training crops "
                    "are missing metadata."
                )

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = self.filenames[idx]

        image_path = (
            self.crop_dir / filename
        )

        if not image_path.exists():
            raise FileNotFoundError(
                f"Crop not found: {image_path}"
            )

        with Image.open(image_path) as image:
            image = image.convert("RGB")
            image = self.transform(image)

        label = float(
            self.label_map[filename]
        )

        return (
            image,
            torch.tensor(
                label,
                dtype=torch.float32,
            ),
        )


def build_dataset(
    dataset_name,
    fraction,
    seed,
    model_name,
):
    """Build a TRAINING dataset."""

    if dataset_name == "nih":
        return NIHDataset(
            model_name=model_name,
            fraction=fraction,
            seed=seed,
            validation=False,
        )

    if dataset_name == "bbbc041":
        return BBBC041Dataset(
            model_name=model_name,
            fraction=fraction,
            seed=seed,
            validation=False,
        )

    raise ValueError(
        f"Unknown dataset: {dataset_name}"
    )


def build_validation_dataset(
    dataset_name,
    model_name,
):
    """Build the FIXED validation dataset."""

    if dataset_name == "nih":
        return NIHDataset(
            model_name=model_name,
            validation=True,
        )

    if dataset_name == "bbbc041":
        return BBBC041Dataset(
            model_name=model_name,
            validation=True,
        )

    raise ValueError(
        f"Unknown dataset: {dataset_name}"
    )