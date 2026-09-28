from data_utils import get_nih_dataset, get_train_test_indices


dataset = get_nih_dataset()

print("Dataset size:", len(dataset))
print("Features:", dataset.features)

print("\nFirst 10 labels:")
print(dataset["label"][:10])

train_indices, test_indices = get_train_test_indices(dataset)

print("\nTrain size:", len(train_indices))
print("Test size:", len(test_indices))

train_labels = [dataset[int(i)]["label"] for i in train_indices]

print("\nTrain label counts:")
print("Parasitized (0):", train_labels.count(0))
print("Uninfected (1):", train_labels.count(1))

test_labels = [dataset[int(i)]["label"] for i in test_indices]

print("\nTest label counts:")
print("Parasitized (0):", test_labels.count(0))
print("Uninfected (1):", test_labels.count(1))