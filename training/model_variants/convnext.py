"""convnext variant — ConvNeXt blocks (depthwise 7x7 + pointwise expansion).

Modernised conv block: a large 7x7 depthwise conv for spatial context, channel
LayerNorm, then a pointwise expand/contract MLP with a residual. Full-res, SiLU.
A strong conv design point against the transformer family.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from model import NUM_FEATURES
from model_variants.heads import SpatialHead


class ConvNeXtBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.dw = nn.Conv2d(
            channels, channels, kernel_size=7, padding=3, groups=channels
        )
        self.norm = nn.GroupNorm(1, channels)  # LayerNorm over channels
        self.pw1 = nn.Conv2d(channels, channels * 4, kernel_size=1)
        self.act = nn.SiLU()
        self.pw2 = nn.Conv2d(channels * 4, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.pw2(self.act(self.pw1(self.norm(self.dw(x)))))
        return x + h


class ConvNeXt(nn.Module):
    def __init__(self, hidden: int = 96, blocks: int = 4) -> None:
        super().__init__()
        self.stem = nn.Conv2d(NUM_FEATURES, hidden, kernel_size=1)
        self.blocks = nn.Sequential(*[ConvNeXtBlock(hidden) for _ in range(blocks)])
        self.head = SpatialHead(hidden)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.head(self.blocks(self.stem(x)))


def build() -> ConvNeXt:
    return ConvNeXt()
