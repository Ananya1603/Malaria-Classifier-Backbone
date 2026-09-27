from datasets import load_dataset
from transformers import AutoImageProcessor, AutoModelForImageClassification
from collections import Counter
import numpy as np


DATASET_ID = "dpdl-benchmark/malaria"
MODEL_ID = "Sadou/malaria-detector-dinov2-tanzania"
SEED = 42


# --------------------------------------------------
# 1. Load NIH malaria dataset
# --------------------------------------------------

print("=" * 60)
print("1. LOADING NIH MALARIA DATASET")
print("=" * 60)

dataset = load_dataset(DATASET_ID)

print("\nDataset:")
print(dataset)

train = dataset["train"]

print(f"\nNumber of training images: {len(train)}")
print(f"Features: {train.features}")


# The dataset stores labels as integers.
# We define the class names explicitly.
label_names = {
    0: "Parasitized",
    1: "Uninfected"
}

print("\nClass labels:")

for label_id, label_name in label_names.items():
    print(f"  {label_id}: {label_name}")


# Full training-set class distribution
full_counts = Counter(train["label"])

print("\nFull training-set class distribution:")

for label_id, count in sorted(full_counts.items()):

    percentage = 100 * count / len(train)

    print(
        f"  {label_id} ({label_names[label_id]}): "
        f"{count} ({percentage:.2f}%)"
    )


# --------------------------------------------------
# 2. Load DINOv2 malaria backbone
# --------------------------------------------------

print("\n" + "=" * 60)
print("2. LOADING DINOv2 MALARIA BACKBONE")
print("=" * 60)

processor = AutoImageProcessor.from_pretrained(MODEL_ID)

model = AutoModelForImageClassification.from_pretrained(
    MODEL_ID
)

print("\nModel loaded successfully.")
print(f"Model type: {model.config.model_type}")

size = processor.size

print(f"Processor size configuration: {size}")

if isinstance(size, dict):

    height = size.get("height")
    width = size.get("width")

    if height is not None and width is not None:
        print(f"Expected input size: {height} x {width}")

    elif "shortest_edge" in size:
        print(
            f"Expected shortest edge: "
            f"{size['shortest_edge']}"
        )

else:
    print(f"Processor size: {size}")


print(f"Number of labels: {model.config.num_labels}")

print("Model labels:")

for label_id, label_name in model.config.id2label.items():
    print(f"  {label_id}: {label_name}")


# --------------------------------------------------
# 3. Stratified subsampling
# --------------------------------------------------

print("\n" + "=" * 60)
print("3. TESTING STRATIFIED SUBSAMPLING")
print("=" * 60)


def get_stratified_subset(dataset, fraction, seed=42):

    """
    Return a stratified subset containing approximately
    `fraction` of the dataset while preserving class ratios.
    """

    if not 0 < fraction <= 1:
        raise ValueError("fraction must be between 0 and 1")

    labels = np.array(dataset["label"])

    rng = np.random.default_rng(seed)

    selected_indices = []

    for label in np.unique(labels):

        class_indices = np.where(labels == label)[0]

        n_samples = max(
            1,
            round(len(class_indices) * fraction)
        )

        chosen = rng.choice(
            class_indices,
            size=n_samples,
            replace=False
        )

        selected_indices.extend(
            chosen.tolist()
        )

    rng.shuffle(selected_indices)

    return dataset.select(selected_indices)


# --------------------------------------------------
# Test 10% subset
# --------------------------------------------------

fraction = 0.10

subset = get_stratified_subset(
    train,
    fraction=fraction,
    seed=SEED
)

print(
    f"\nRequested fraction: "
    f"{fraction * 100:.0f}%"
)

print(
    f"Full training set: "
    f"{len(train)} images"
)

print(
    f"Subset: "
    f"{len(subset)} images"
)


subset_counts = Counter(subset["label"])

print("\n10% subset class distribution:")

for label_id, count in sorted(subset_counts.items()):

    percentage = 100 * count / len(subset)

    print(
        f"  {label_id} "
        f"({label_names[label_id]}): "
        f"{count} ({percentage:.2f}%)"
    )


# --------------------------------------------------
# Verify stratification
# --------------------------------------------------

print("\nClass-ratio verification:")

for label_id in sorted(full_counts):

    full_ratio = (
        full_counts[label_id] /
        len(train)
    )

    subset_ratio = (
        subset_counts[label_id] /
        len(subset)
    )

    print(
        f"  {label_names[label_id]}: "
        f"full={full_ratio:.4f}, "
        f"subset={subset_ratio:.4f}"
    )


# --------------------------------------------------
# Test all four label fractions
# --------------------------------------------------

print("\n" + "=" * 60)
print("4. TESTING ALL LABEL FRACTIONS")
print("=" * 60)

for fraction in [1.0, 0.25, 0.10, 0.02]:

    subset = get_stratified_subset(
        train,
        fraction=fraction,
        seed=SEED
    )

    print(
        f"{fraction * 100:.0f}% -> "
        f"{len(subset)} images"
    )


print("\n" + "=" * 60)
print("MILESTONE 1 SETUP CHECK COMPLETE")
print("=" * 60)
