import torch
import torch.nn as nn


class SmallCNN(nn.Module):
    """
    Small CNN baseline for Milestone 4.

    Input:
        RGB image tensor [B, 3, 80, 80]

    Architecture:
        Conv2d(3 -> 16, 3x3)
        ReLU
        MaxPool2d(2x2)

        Conv2d(16 -> 32, 3x3)
        ReLU
        MaxPool2d(2x2)

        Conv2d(32 -> 64, 3x3)
        ReLU
        MaxPool2d(2x2)

        Flatten
        Linear(4096 -> 128)
        ReLU
        Linear(128 -> 1)

    Output:
        One raw logit per image: [B]

    IMPORTANT:
        No sigmoid is included because training uses
        BCEWithLogitsLoss.
    """

    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=16,
                kernel_size=3,
            ),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=3,
            ),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
            ),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(4096, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)

        # [B, 1] -> [B]
        return x.squeeze(1)


def build_small_cnn():
    """
    Return a NEW randomly initialized Small CNN.

    Every experiment must call this independently.
    No weights are loaded from another run.
    """
    return SmallCNN()


if __name__ == "__main__":
    model = build_small_cnn()

    x = torch.randn(4, 3, 80, 80)

    with torch.no_grad():
        y = model(x)

    print(model)
    print()
    print("Input shape :", tuple(x.shape))
    print("Output shape:", tuple(y.shape))
    print(
        "Parameters:",
        sum(p.numel() for p in model.parameters()),
    )

    assert y.shape == (4,), (
        f"Expected output shape (4,), got {tuple(y.shape)}"
    )

    print("SMALL CNN SHAPE TEST: PASSED")
