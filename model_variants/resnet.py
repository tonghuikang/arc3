"""resnet variant — residual 3x3 conv blocks at full resolution.

Classic pre-activation residual stack (GroupNorm + GELU). Residual connections
let it go deeper than the plain ``deep``/``deeper`` MLP variants without
vanishing gradients, while 3x3 convs give it the local spatial context the 1x1
per-pixel variants lack.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from model import NUM_FEATURES
from model_variants.heads import SpatialHead


class ResBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(8, channels)
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(8, channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.gelu(self.norm1(x)))
        h = self.conv2(F.gelu(self.norm2(h)))
        return x + h


class ResNetPolicy(nn.Module):
    def __init__(self, hidden: int = 96, blocks: int = 5) -> None:
        super().__init__()
        self.stem = nn.Conv2d(NUM_FEATURES, hidden, kernel_size=3, padding=1)
        self.blocks = nn.Sequential(*[ResBlock(hidden) for _ in range(blocks)])
        self.head = SpatialHead(hidden)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.head(self.blocks(self.stem(x)))


def build() -> ResNetPolicy:
    return ResNetPolicy()
