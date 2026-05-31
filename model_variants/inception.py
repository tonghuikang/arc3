"""inception variant — multi-scale parallel convolutions per block.

Each block runs four parallel branches over the same input — 1x1, 3x3, dilated
3x3 (≈5x5 reach), and pooled-then-1x1 — and concatenates them, so a single layer
mixes several receptive-field scales at once. Full-res, SiLU.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from model import NUM_FEATURES
from model_variants.heads import SpatialHead


class InceptionBlock(nn.Module):
    def __init__(self, in_c: int, out_c: int) -> None:
        super().__init__()
        q = out_c // 4
        self.b1 = nn.Conv2d(in_c, q, kernel_size=1)
        self.b3 = nn.Conv2d(in_c, q, kernel_size=3, padding=1)
        self.b5 = nn.Conv2d(in_c, q, kernel_size=3, padding=2, dilation=2)
        self.bp = nn.Sequential(
            nn.AvgPool2d(3, stride=1, padding=1),
            nn.Conv2d(in_c, out_c - 3 * q, kernel_size=1),
        )
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = torch.cat([self.b1(x), self.b3(x), self.b5(x), self.bp(x)], dim=1)
        return self.act(y)


class Inception(nn.Module):
    def __init__(self, hidden: int = 96, blocks: int = 3) -> None:
        super().__init__()
        mods: list[nn.Module] = []
        in_c = NUM_FEATURES
        for _ in range(blocks):
            mods.append(InceptionBlock(in_c, hidden))
            in_c = hidden
        self.body = nn.Sequential(*mods)
        self.head = SpatialHead(hidden)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.head(self.body(x))


def build() -> Inception:
    return Inception()
