"""van — Visual Attention Network (Large Kernel Attention).

LKA factorises a large-kernel attention into a depthwise conv + a dilated
depthwise conv (huge effective receptive field) + a pointwise, used as a
multiplicative attention gate. Kept inside pre-normed residual blocks so the
gating stays stable (unlike the naive main-path SwiGLU that diverged as
`gated_conv`). Width 128, full-res, SiLU.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from model import NUM_FEATURES
from model_variants.heads import SpatialHead


class LKA(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dw = nn.Conv2d(dim, dim, kernel_size=5, padding=2, groups=dim)
        self.dwd = nn.Conv2d(
            dim, dim, kernel_size=7, stride=1, padding=9, groups=dim, dilation=3
        )
        self.pw = nn.Conv2d(dim, dim, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn = self.pw(self.dwd(self.dw(x)))
        return x * attn


class VANBlock(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(1, dim)
        self.proj1 = nn.Conv2d(dim, dim, kernel_size=1)
        self.act = nn.SiLU()
        self.lka = LKA(dim)
        self.proj2 = nn.Conv2d(dim, dim, kernel_size=1)
        self.norm2 = nn.GroupNorm(1, dim)
        self.mlp1 = nn.Conv2d(dim, dim * 4, kernel_size=1)
        self.mlp2 = nn.Conv2d(dim * 4, dim, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.proj2(self.lka(self.act(self.proj1(self.norm1(x)))))
        x = x + h
        h = self.mlp2(self.act(self.mlp1(self.norm2(x))))
        return x + h


class VAN(nn.Module):
    def __init__(self, hidden: int = 128, blocks: int = 4) -> None:
        super().__init__()
        self.stem = nn.Conv2d(NUM_FEATURES, hidden, kernel_size=1)
        self.blocks = nn.Sequential(*[VANBlock(hidden) for _ in range(blocks)])
        self.head = SpatialHead(hidden)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.head(self.blocks(self.stem(x)))


def build() -> VAN:
    return VAN()
