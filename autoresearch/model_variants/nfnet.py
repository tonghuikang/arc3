"""nfnet — normalizer-free residual blocks (NFNet-style).

No normalization layers at all. Each block scales its pre-activation input by
1/beta and its residual branch output by a small constant alpha, the
signal-propagation trick that lets deep residual nets train stably without
BatchNorm/GroupNorm. Probes whether the GroupNorm in the conv winners is load-
bearing or removable. Width 128, full-res, SiLU.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from model import NUM_FEATURES
from model_variants.heads import SpatialHead


class NFBlock(nn.Module):
    def __init__(self, channels: int, alpha: float = 0.2, beta: float = 1.0) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.act = nn.SiLU()
        self.alpha = alpha
        self.beta = beta

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.act(x) / self.beta
        h = self.conv2(self.act(self.conv1(h)))
        return x + self.alpha * h


class NFNet(nn.Module):
    def __init__(self, hidden: int = 128, blocks: int = 4) -> None:
        super().__init__()
        self.stem = nn.Conv2d(NUM_FEATURES, hidden, kernel_size=3, padding=1)
        self.blocks = nn.Sequential(*[NFBlock(hidden) for _ in range(blocks)])
        self.head = SpatialHead(hidden)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.head(self.blocks(self.stem(x)))


def build() -> NFNet:
    return NFNet()
