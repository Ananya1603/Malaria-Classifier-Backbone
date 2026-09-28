import os
import numpy as np

from datasets import load_dataset
from sklearn.model_selection import train_test_split


def get_nih_dataset():
    dataset = load_dataset("dpdl-benchmark/malaria")
    return dataset["train"]


def get_train_test_indices(dataset, test_size=0.2, seed=42):

    indices_path = "test_indices.npy"

    if os.path.exists(indices_path):
        print(f"Loading existing test indices from {indices_path}")

        test_indices = np.load(indices_path)
        test_indices = np.asarray(test_indices, dtype=int)

        all_indices = np.arange(len(dataset))

        train_indices = np.setdiff1d(
            all_indices,
            test_indices
        )

    else:
        print("Generating new 80/20 stratified split...")

        labels = np.asarray(dataset["label"])
        indices = np.arange(len(dataset))

        train_indices, test_indices = train_test_split(
            indices,
            test_size=test_size,
            random_state=seed,
            stratify=labels
        )

        train_indices = np.asarray(train_indices, dtype=int)
        test_indices = np.asarray(test_indices, dtype=int)

        np.save(indices_path, test_indices)

        print(f"Saved test indices to {indices_path}")

    return train_indices, test_indices


def get_stratified_subset(dataset, indices, fraction, seed=42):

    indices = np.asarray(indices, dtype=int)

    if fraction == 1.0:
        return indices

    labels = np.asarray(
        [dataset[int(i)]["label"] for i in indices]
    )

    _, subset_indices = train_test_split(
        indices,
        test_size=fraction,
        random_state=seed,
        stratify=labels
    )

    return np.asarray(subset_indices, dtype=int)