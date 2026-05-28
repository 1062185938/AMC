import torch
from torch import nn


class SimpleCNN1D(nn.Module):
    def __init__(self, num_classes=11, dropout=0.3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(2, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        if x.ndim != 3 or x.shape[1] != 2 or x.shape[2] != 128:
            raise ValueError(f"Expected input shape [B, 2, 128], got {tuple(x.shape)}")
        x = self.features(x)
        return self.classifier(x)


if __name__ == "__main__":
    model = SimpleCNN1D()
    dummy = torch.randn(4, 2, 128)
    logits = model(dummy)
    print(logits.shape)
