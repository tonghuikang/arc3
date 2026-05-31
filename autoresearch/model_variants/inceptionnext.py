"""inceptionnext — InceptionNeXt blocks (multi-shape depthwise mixing).

Splits channels four ways and applies different depthwise kernels in parallel:
identity, a 3x3 square, a 1xK horizontal band, and a Kx1 vertical band. This
captures square + axis-aligned spatial structure cheaply — a good prior for
grid/board tasks. No multiplicative gating. Width 128, band=11, full-res, SiLU.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from model import NUM_FEATURES
from model_variants.heads import SpatialHead


class InceptionDW(nn.Module):
    def __init__(self, dim: int, band: int = 11) -> None:
        super().__init__()
        g = dim // 4
        self.g = g
        self.id_split = dim - 3 * g
        self.sq = nn.Conv2d(g, g, kernel_size=3, padding=1, groups=g)
        self.hw = nn.Conv2d(
            g, g, kernel_size=(1, band), padding=(0, band // 2), groups=g
        )
        self.vw = nn.Conv2d(
            g, g, kernel_size=(band, 1), padding=(band // 2, 0), groups=g
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_id, x_sq, x_h, x_v = torch.split(
            x, [self.id_split, self.g, self.g, self.g], dim=1
        )
        return torch.cat([x_id, self.sq(x_sq), self.hw(x_h), self.vw(x_v)], dim=1)


class InceptionNeXtBlock(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dw = InceptionDW(dim)
        self.norm = nn.GroupNorm(1, dim)
        self.pw1 = nn.Conv2d(dim, dim * 4, kernel_size=1)
        self.act = nn.SiLU()
        self.pw2 = nn.Conv2d(dim * 4, dim, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.pw2(self.act(self.pw1(self.norm(self.dw(x)))))
        return x + h


class InceptionNeXt(nn.Module):
    def __init__(self, hidden: int = 128, blocks: int = 4) -> None:
        super().__init__()
        self.stem = nn.Conv2d(NUM_FEATURES, hidden, kernel_size=1)
        self.blocks = nn.Sequential(
            *[InceptionNeXtBlock(hidden) for _ in range(blocks)]
        )
        self.head = SpatialHead(hidden)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.head(self.blocks(self.stem(x)))


def build() -> InceptionNeXt:
    return InceptionNeXt()
