"""ghost — GhostNet-style cheap-feature blocks.

A Ghost module generates half its output channels with a normal 1x1 conv and the
other half with a cheap depthwise 3x3 conv applied to those primary features,
then concatenates. Two Ghost modules per residual block. Tests whether redundant
feature maps can be produced cheaply without hurting the policy heads. Width 128,
full-res, SiLU.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from model import NUM_FEATURES
from model_variants.heads import SpatialHead


class GhostModule(nn.Module):
    def __init__(self, in_c: int, out_c: int) -> None:
        super().__init__()
        assert out_c % 2 == 0, "out_c must be even"
        init_c = out_c // 2
        self.primary = nn.Conv2d(in_c, init_c, kernel_size=1)
        self.cheap = nn.Conv2d(init_c, init_c, kernel_size=3, padding=1, groups=init_c)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        p = self.act(self.primary(x))
        c = self.act(self.cheap(p))
        return torch.cat([p, c], dim=1)


class GhostBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.norm = nn.GroupNorm(8, channels)
        self.g1 = GhostModule(channels, channels)
        self.g2 = GhostModule(channels, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.g2(self.g1(self.norm(x)))


class GhostNet(nn.Module):
    def __init__(self, hidden: int = 128, blocks: int = 4) -> None:
        super().__init__()
        self.stem = nn.Conv2d(NUM_FEATURES, hidden, kernel_size=1)
        self.blocks = nn.Sequential(*[GhostBlock(hidden) for _ in range(blocks)])
        self.head = SpatialHead(hidden)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.head(self.blocks(self.stem(x)))


def build() -> GhostNet:
    return GhostNet()
