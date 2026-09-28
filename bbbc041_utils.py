import json
import csv
from pathlib import Path
from collections import Counter

import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split


DATA_DIR = Path("data/bbbc041/malaria")
IMAGE_DIR = DATA_DIR / "images"
OUTPUT_DIR = Path("data/bbbc041_processed")

SEED = 42
TEST_SIZE = 0.20

# Canonical binary mapping shared with NIH:
# 0 = Parasitized
# 1 = Uninfected
LABEL_MAP = {
    "ring": 0,
    "trophozoite": 0,
    "schizont": 0,
    "gametocyte": 0,
    "red blood cell": 1,
}

EXCLUDED_CATEGORIES = {
    "leukocyte",
    "difficult",
}


def load_records():
    """Load all annotated BBBC041 source images."""
    records = []

    for filename in ["training.json", "test.json"]:
        with open(DATA_DIR / filename) as f:
            records.extend(json.load(f))

    return records


def get_image_filename(record):
    """Extract filename from BBBC041 image metadata."""
    return Path(record["image"]["pathname"]).name


def get_image_stratum(record):
    """
    Stratify source images according to whether they contain
    at least one annotated malaria-infected cell.
    """
    infected_categories = {
        "ring",
        "trophozoite",
        "schizont",
        "gametocyte",
    }

    categories = {
        obj.get("category")
        for obj in record["objects"]
    }

    return int(bool(categories & infected_categories))


def create_image_split(records):
    """
    Create fixed 80/20 SOURCE-IMAGE split.

    Cells are NOT split independently.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    train_file = OUTPUT_DIR / "train_image_ids.npy"
    test_file = OUTPUT_DIR / "test_image_ids.npy"

    image_ids = np.array([
        get_image_filename(record)
        for record in records
    ])

    strata = np.array([
        get_image_stratum(record)
        for record in records
    ])

    if train_file.exists() and test_file.exists():
        print("Loading existing BBBC041 image-level split.")

        train_ids = np.load(train_file, allow_pickle=True)
        test_ids = np.load(test_file, allow_pickle=True)

    else:
        print("Generating fixed 80/20 BBBC041 image-level split...")

        train_ids, test_ids = train_test_split(
            image_ids,
            test_size=TEST_SIZE,
            random_state=SEED,
            stratify=strata,
        )

        np.save(train_file, train_ids)
        np.save(test_file, test_ids)

        print("Saved fixed image-level split.")

    return train_ids, test_ids


def crop_split(records, image_ids, split_name):
    """
    Extract single-cell crops for one source-image split.
    """
    split_dir = OUTPUT_DIR / split_name
    split_dir.mkdir(parents=True, exist_ok=True)

    metadata_path = OUTPUT_DIR / f"{split_name}_metadata.csv"

    selected_ids = set(image_ids.tolist())

    rows = []
    counts = Counter()
    crop_number = 0

    for record in records:
        source_image = get_image_filename(record)

        if source_image not in selected_ids:
            continue

        image_path = IMAGE_DIR / source_image

        if not image_path.exists():
            print(f"WARNING: Missing image: {image_path}")
            continue

        with Image.open(image_path) as img:
            img = img.convert("RGB")

            width, height = img.size

            for object_index, obj in enumerate(record["objects"]):
                category = obj.get("category")

                if category not in LABEL_MAP:
                    continue

                bbox = obj["bounding_box"]

                # BBBC041 uses:
                # r = row (y)
                # c = column (x)
                r_min = int(bbox["minimum"]["r"])
                c_min = int(bbox["minimum"]["c"])
                r_max = int(bbox["maximum"]["r"])
                c_max = int(bbox["maximum"]["c"])

                # Clamp coordinates to image bounds.
                left = max(0, c_min)
                upper = max(0, r_min)
                right = min(width, c_max)
                lower = min(height, r_max)

                if right <= left or lower <= upper:
                    continue

                crop = img.crop(
                    (left, upper, right, lower)
                )

                label = LABEL_MAP[category]

                crop_filename = (
                    f"{Path(source_image).stem}"
                    f"__cell_{object_index:04d}"
                    f"__label_{label}.png"
                )

                crop_path = split_dir / crop_filename
                crop.save(crop_path)

                rows.append({
                    "crop_filename": crop_filename,
                    "source_image": source_image,
                    "object_index": object_index,
                    "original_category": category,
                    "label": label,
                    "r_min": r_min,
                    "c_min": c_min,
                    "r_max": r_max,
                    "c_max": c_max,
                })

                counts[category] += 1
                crop_number += 1

    with open(metadata_path, "w", newline="") as f:
        fieldnames = [
            "crop_filename",
            "source_image",
            "object_index",
            "original_category",
            "label",
            "r_min",
            "c_min",
            "r_max",
            "c_max",
        ]

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(rows)

    print(f"\n{split_name.upper()}")
    print(f"Created {crop_number} crops")

    for category, count in sorted(counts.items()):
        print(f"  {category}: {count}")

    print(f"Metadata: {metadata_path}")


def verify_no_leakage(train_ids, test_ids):
    """Ensure no source image occurs in both splits."""
    train_set = set(train_ids.tolist())
    test_set = set(test_ids.tolist())

    overlap = train_set & test_set

    if overlap:
        raise RuntimeError(
            f"DATA LEAKAGE: {len(overlap)} images occur "
            "in both train and test!"
        )

    print("\nLeakage check PASSED.")
    print("Source images shared between train/test: 0")


def main():
    print("Loading BBBC041 annotations...")

    records = load_records()

    print(f"Total source images: {len(records)}")

    train_ids, test_ids = create_image_split(records)

    print(f"Train source images: {len(train_ids)}")
    print(f"Test source images: {len(test_ids)}")

    verify_no_leakage(train_ids, test_ids)

    print("\nExtracting TRAIN crops...")
    crop_split(
        records,
        train_ids,
        "train"
    )

    print("\nExtracting TEST crops...")
    crop_split(
        records,
        test_ids,
        "test"
    )

    print("\nBBBC041 processing complete.")


if __name__ == "__main__":
    main()