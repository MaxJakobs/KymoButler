"""Simple classification network.

Corresponds to classnet in NeuralNetworkDefs.wl lines 52-58.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from kymobutler.models.unet import BasicBlock


class ClassNet(nn.Module):
    """Simple CNN classifier with 4 pooling stages.

    Used for auxiliary classification tasks in KymoButler.
    """

    def __init__(self, n_classes: int, input_size: int, n: int = 64):
        super().__init__()
        final_pool_size = math.ceil(input_size / 16)
        self.features = nn.Sequential(
            BasicBlock(1, n),
            BasicBlock(n, n),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(),
            BasicBlock(n, 2 * n),
            BasicBlock(2 * n, 2 * n),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(),
            BasicBlock(2 * n, 4 * n),
            BasicBlock(4 * n, 4 * n),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(),
            nn.AdaptiveAvgPool2d(final_pool_size),
            nn.Dropout2d(),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(4 * n * final_pool_size * final_pool_size, 256),
            nn.LeakyReLU(0.1),
            nn.Linear(256, n_classes),
            nn.Softmax(dim=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))
