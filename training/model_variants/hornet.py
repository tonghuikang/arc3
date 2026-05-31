"""hornet — recursive gated convolutions (HorNet gnconv).

gnconv realises high-order spatial interactions: a pointwise projection splits
into a base path and gating paths, and a depthwise conv output multiplicatively
gates the base across recursive orders. Unlike a naive SwiGLU on the main path
(which diverged here as `gated_conv`), this keeps the gating *inside* a
pre-normed residual block with a 1/order scale damp, the canonical stable form.
Order 2, width 128, full-res, SiLU.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from model import NUM_FEATURES
from model_variants.heads import SpatialHead


class GnConv(nn.Module):
    def __init__(self, dim: int, order: int = 2) -> None:
        super().__init__()
        self.order = order
        self.proj_in = nn.Conv2d(dim, 2 * dim, kernel_size=1)
        self.dw = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.proj_out = nn.Conv2d(dim, dim, kernel_size=1)
        self.scale = 1.0 / order  # damp the multiplicative gating

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dim = x.shape[1]
        base, gate = torch.split(self.proj_in(x), [dim, dim], dim=1)
        gated = base * (self.dw(gate) * self.scale)  # 2nd-order interaction
        return self.proj_out(gated)


class HorNetBlock(nn.Module):
    def __init__(self, dim: int, expansion: int = 4) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(1, dim)
        self.gnconv = GnConv(dim)
        self.norm2 = nn.GroupNorm(1, dim)
        self.pw1 = nn.Conv2d(dim, dim * expansion, kernel_size=1)
        self.act = nn.SiLU()
        self.pw2 = nn.Conv2d(dim * expansion, dim, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.gnconv(self.norm1(x))
        x = x + self.pw2(self.act(self.pw1(self.norm2(x))))
        return x


class HorNet(nn.Module):
    def __init__(self, hidden: int = 128, blocks: int = 4) -> None:
        super().__init__()
        self.stem = nn.Conv2d(NUM_FEATURES, hidden, kernel_size=1)
        self.blocks = nn.Sequential(*[HorNetBlock(hidden) for _ in range(blocks)])
        self.head = SpatialHead(hidden)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.head(self.blocks(self.stem(x)))


def build() -> HorNet:
    return HorNet()
