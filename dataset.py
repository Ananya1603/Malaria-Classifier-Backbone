import csv
import json
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset

from data_utils import get_nih_dataset


NIH_SUBSET_DIR = Path("subsets/nih")
BBBC_SUBSET_DIR = Path("subsets/bbbc041")
VAL_DIR = Path("data/validation")

BBBC_IMAGE_DIR = Path("data/bbbc041_processed/train")


def _fraction_name(fraction):
    fraction = int(fraction)
    if fraction not in (2, 10, 25, 100):
        raise ValueError("fraction must be one of: 2, 10, 25, 100")
    return fraction


def _load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _nih_subset_path(fraction, seed):
    fraction = _fraction_name(fraction)

    if fraction == 100:
        return NIH_SUBSET_DIR / "fraction_100.json"

    return NIH_SUBSET_DIR / f"fraction_{fraction}_seed_{seed}.json"


def _bbbc_subset_path(fraction, seed):
    fraction = _fraction_name(fraction)

    if fraction == 100:
        return BBBC_SUBSET_DIR / "fraction_100.csv"

    return BBBC_SUBSET_DIR / f"fraction_{fraction}_seed_{seed}.csv"


class NIHMalariaDataset(Dataset):
    """
    NIH malaria dataset.

    Original NIH labels:
        0 = Parasitized
        1 = Uninfected

    Binary training target used in Milestone 4:
        1 = malaria positive / parasitized
        0 = malaria negative / uninfected
    """

    def __init__(self, indices, processor):
        self.dataset = get_nih_dataset()
        self.indices = [int(x) for x in indices]
        self.processor = processor

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        row = self.dataset[self.indices[idx]]

        image = row["image"].convert("RGB")

        # NIH: 0=Parasitized, 1=Uninfected.
        # Convert to target 1=malaria-positive.
        target = 1.0 if int(row["label"]) == 0 else 0.0

        pixel_values = self.processor(
            images=image,
            return_tensors="pt",
        )["pixel_values"].squeeze(0)

        return pixel_values, target


class BBBC041Dataset(Dataset):
    """
    BBBC041 cropped-cell dataset.

    Existing metadata:
        label 1 = red blood cell / uninfected
        label 0 = malaria-infected cell

    Binary training target used in Milestone 4:
        1 = malaria positive
        0 = malaria negative
    """

    def __init__(self, rows, processor):
        self.rows = rows
        self.processor = processor

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]

        image_path = BBBC_IMAGE_DIR / row["crop_filename"]

        if not image_path.exists():
            raise FileNotFoundError(f"Missing BBBC041 crop: {image_path}")

        with Image.open(image_path) as img:
            image = img.convert("RGB")

        raw_label = int(row["label"])

        # Existing BBBC metadata:
        # 1 = red blood cell (negative)
        # 0 = infected category (positive)
        target = 1.0 if raw_label == 0 else 0.0

        pixel_values = self.processor(
            images=image,
            return_tensors="pt",
        )["pixel_values"].squeeze(0)

        return pixel_values, target


def load_nih_splits(fraction, seed, processor):
    """
    Load a Milestone 3 NIH subset and intersect it with the fixed
    post-validation training pool.

    Validation samples are NEVER allowed into the training dataset.
    """

    subset_path = _nih_subset_path(fraction, seed)

    if not subset_path.exists():
        raise FileNotFoundError(subset_path)

    subset_indices = [int(x) for x in _load_json(subset_path)]

    train_pool_path = VAL_DIR / "nih_train_indices.json"
    val_path = VAL_DIR / "nih_val_indices.json"

    train_pool = set(int(x) for x in _load_json(train_pool_path))
    val_indices = [int(x) for x in _load_json(val_path)]

    # Preserve the existing Milestone 3 subset; only remove examples
    # assigned to the subsequently-created fixed validation set.
    filtered_train = [x for x in subset_indices if x in train_pool]

    overlap = set(filtered_train) & set(val_indices)
    if overlap:
        raise RuntimeError(
            f"NIH train/validation leakage detected: {len(overlap)} examples"
        )

    train_dataset = NIHMalariaDataset(filtered_train, processor)
    val_dataset = NIHMalariaDataset(val_indices, processor)

    return train_dataset, val_dataset


def _read_csv(path):
    with open(path, "r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _read_bbbc_subset_filenames(path):
    rows = _read_csv(path)

    if not rows:
        return []

    # Milestone 3 CSVs may use crop_filename or filename.
    if "crop_filename" in rows[0]:
        key = "crop_filename"
    elif "filename" in rows[0]:
        key = "filename"
    else:
        raise ValueError(
            f"Could not find crop filename column in {path}. "
            f"Columns: {list(rows[0].keys())}"
        )

    return [row[key] for row in rows]


def load_bbbc_splits(fraction, seed, processor):
    """
    Load a Milestone 3 BBBC041 subset and intersect it with the fixed
    post-validation training metadata.

    Validation separation remains at source-image level.
    """

    subset_path = _bbbc_subset_path(fraction, seed)

    if not subset_path.exists():
        raise FileNotFoundError(subset_path)

    subset_filenames = set(_read_bbbc_subset_filenames(subset_path))

    train_metadata_path = VAL_DIR / "bbbc041_train_metadata.csv"
    val_metadata_path = VAL_DIR / "bbbc041_val_metadata.csv"

    train_rows_all = _read_csv(train_metadata_path)
    val_rows = _read_csv(val_metadata_path)

    # Preserve the Milestone 3 subset and remove anything belonging
    # to the fixed validation source-image split.
    train_rows = [
        row
        for row in train_rows_all
        if row["crop_filename"] in subset_filenames
    ]

    train_names = {row["crop_filename"] for row in train_rows}
    val_names = {row["crop_filename"] for row in val_rows}

    overlap = train_names & val_names
    if overlap:
        raise RuntimeError(
            f"BBBC041 train/validation crop leakage detected: {len(overlap)}"
        )

    train_sources = {row["source_image"] for row in train_rows}
    val_sources = {row["source_image"] for row in val_rows}

    source_overlap = train_sources & val_sources
    if source_overlap:
        raise RuntimeError(
            "BBBC041 source-image leakage detected: "
            f"{len(source_overlap)} source images"
        )

    train_dataset = BBBC041Dataset(train_rows, processor)
    val_dataset = BBBC041Dataset(val_rows, processor)

    return train_dataset, val_dataset


def count_binary_targets(dataset):
    """
    Count Milestone 4 binary targets.

    Returns:
        negative_count: malaria-negative examples (target 0)
        positive_count: malaria-positive examples (target 1)
    """

    if isinstance(dataset, NIHMalariaDataset):
        positive = 0
        negative = 0

        for index in dataset.indices:
            raw_label = int(dataset.dataset[index]["label"])
            if raw_label == 0:
                positive += 1
            else:
                negative += 1

        return negative, positive

    if isinstance(dataset, BBBC041Dataset):
        positive = sum(int(row["label"]) == 0 for row in dataset.rows)
        negative = sum(int(row["label"]) == 1 for row in dataset.rows)

        return negative, positive

    raise TypeError(f"Unsupported dataset type: {type(dataset)}")