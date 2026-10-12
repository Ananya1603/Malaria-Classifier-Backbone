import torch
import torch.nn as nn


class SmallCNN(nn.Module):
    """
    Small CNN v2 baseline.

    Input:
        RGB image, 80 x 80

    Architecture:
        Conv2d(3 -> 16, 3x3)
        MaxPool2d(2x2)
        Conv2d(16 -> 32, 3x3)
        MaxPool2d(2x2)
        Conv2d(32 -> 64, 3x3)
        MaxPool2d(2x2)
        Flatten
        Linear -> 128
        ReLU
        Linear -> 1 logit
    """

    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(16, 32, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(32, 64, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x.squeeze(1)


def build_small_cnn():
    """Small CNN initialized completely from scratch."""
    return SmallCNN()


def build_resnet18():
    """
    ImageNet-pretrained ResNet-18 with a new binary
    classification head.
    """
    from torchvision.models import resnet18, ResNet18_Weights

    model = resnet18(weights=ResNet18_Weights.DEFAULT)

    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 1)

    return model


def build_dinov2(checkpoint="Sadou/malaria-detector-dinov2-tanzania"):
    """
    Load the malaria DINOv2 checkpoint and replace its
    classification head with a one-logit binary head.

    The complete model remains trainable for full fine-tuning.
    """
    from transformers import AutoModelForImageClassification

    model = AutoModelForImageClassification.from_pretrained(
        checkpoint
    )

    num_features = model.classifier.in_features
    model.classifier = nn.Linear(num_features, 1)

    return model


def build_plasmovision(
    checkpoint="ngohjuniormbah/plasmovision-malaria-ai"
):
    """
    Load the PlasmoVision checkpoint and replace its
    classification head with a one-logit binary head.

    The complete model remains trainable for full fine-tuning.
    """
    from transformers import AutoModelForImageClassification

    model = AutoModelForImageClassification.from_pretrained(
        checkpoint
    )

    num_features = model.classifier.in_features
    model.classifier = nn.Linear(num_features, 1)

    return model


def build_model(name):
    """
    Common model factory used by the training script.
    """

    name = name.lower()

    if name == "small_cnn":
        return build_small_cnn()

    if name == "resnet18":
        return build_resnet18()

    if name == "dinov2":
        return build_dinov2()

    if name == "plasmovision":
        return build_plasmovision()

    raise ValueError(f"Unknown model: {name}")