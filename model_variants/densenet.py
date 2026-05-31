"""densenet variant — densely-connected conv layers.

Each layer's 3x3 conv sees the concatenation of all previous layers' features
(feature reuse / dense connectivity), then a 1x1 projection feeds the head.
Different connectivity pattern from the residual stacks. Full-res, SiLU.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from model import NUM_FEATURES
from model_variants.heads import SpatialHead


class DenseLayer(nn.Module):
    def __init__(self, in_c: int, growth: int) -> None:
        super().__init__()
        self.norm = nn.GroupNorm(1, in_c)
        self.act = nn.SiLU()
        self.conv = nn.Conv2d(in_c, growth, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([x, self.conv(self.act(self.norm(x)))], dim=1)


class DenseNet(nn.Module):
    def __init__(self, growth: int = 32, layers: int = 5, out: int = 96) -> None:
        super().__init__()
        self.stem = nn.Conv2d(NUM_FEATURES, 64, kernel_size=1)
        mods: list[nn.Module] = []
        in_c = 64
        for _ in range(layers):
            mods.append(DenseLayer(in_c, growth))
            in_c += growth
        self.block = nn.Sequential(*mods)
        self.proj = nn.Conv2d(in_c, out, kernel_size=1)
        self.head = SpatialHead(out)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.head(self.proj(self.block(self.stem(x))))


def build() -> DenseNet:
    return DenseNet()
