"""bottleneck variant — bottleneck residual conv blocks (1x1 -> 3x3 -> 1x1).

ResNet-50-style bottleneck: squeeze channels with 1x1, do the 3x3 spatial conv in
the reduced space, expand back with 1x1, add residual. Cheaper deep stacks than
plain 3x3 residual blocks. Full-res, GroupNorm, SiLU.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from model import NUM_FEATURES
from model_variants.heads import SpatialHead


class BottleneckBlock(nn.Module):
    def __init__(self, channels: int, mid: int) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(8, channels)
        self.conv1 = nn.Conv2d(channels, mid, kernel_size=1)
        self.norm2 = nn.GroupNorm(8, mid)
        self.conv2 = nn.Conv2d(mid, mid, kernel_size=3, padding=1)
        self.norm3 = nn.GroupNorm(8, mid)
        self.conv3 = nn.Conv2d(mid, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = self.conv2(F.silu(self.norm2(h)))
        h = self.conv3(F.silu(self.norm3(h)))
        return x + h


class Bottleneck(nn.Module):
    def __init__(self, hidden: int = 128, mid: int = 32, blocks: int = 4) -> None:
        super().__init__()
        self.stem = nn.Conv2d(NUM_FEATURES, hidden, kernel_size=3, padding=1)
        self.blocks = nn.Sequential(
            *[BottleneckBlock(hidden, mid) for _ in range(blocks)]
        )
        self.head = SpatialHead(hidden)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.head(self.blocks(self.stem(x)))


def build() -> Bottleneck:
    return Bottleneck()
