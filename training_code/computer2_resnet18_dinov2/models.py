import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


def build_resnet18():
    """
    Milestone 4 ResNet-18.

    Initialization:
        ImageNet IMAGENET1K_V1

    Classification head:
        Fresh single-logit binary head.

    Training:
        Entire backbone + head remain trainable.
    """

    weights = ResNet18_Weights.IMAGENET1K_V1

    model = resnet18(weights=weights)

    in_features = model.fc.in_features

    # Replace ImageNet 1000-class classifier with fresh binary head.
    model.fc = nn.Linear(in_features, 1)

    # Explicitly ensure full-network fine-tuning.
    for parameter in model.parameters():
        parameter.requires_grad = True

    return model, weights
