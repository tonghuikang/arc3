"""starnet — StarNet "star operation" blocks (element-wise multiplied projections).

The star op f = act(W1 x) * (W2 x) implicitly maps to a very high-dimensional
feature space without widening, a cheap way to add representational power. Gating
is kept inside a pre-normed residual block (the naive main-path SwiGLU diverged
here as `gated_conv`; this is the stable StarNet form). Width 128, full-res, SiLU.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from model import NUM_FEATURES
from model_variants.heads import SpatialHead


class StarBlock(nn.Module):
    def __init__(self, dim: int, expansion: int = 4) -> None:
        super().__init__()
        self.dw = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = nn.GroupNorm(1, dim)
        self.f1 = nn.Conv2d(dim, dim * expansion, kernel_size=1)
        self.f2 = nn.Conv2d(dim, dim * expansion, kernel_size=1)
        self.act = nn.SiLU()
        self.g = nn.Conv2d(dim * expansion, dim, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(self.dw(x))
        h = self.act(self.f1(h)) * self.f2(h)  # star operation
        return x + self.g(h)


class StarNet(nn.Module):
    def __init__(self, hidden: int = 128, blocks: int = 4) -> None:
        super().__init__()
        self.stem = nn.Conv2d(NUM_FEATURES, hidden, kernel_size=1)
        self.blocks = nn.Sequential(*[StarBlock(hidden) for _ in range(blocks)])
        self.head = SpatialHead(hidden)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.head(self.blocks(self.stem(x)))


def build() -> StarNet:
    return StarNet()
