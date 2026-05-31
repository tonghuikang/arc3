"""convmixer variant — ConvMixer (depthwise large-kernel + pointwise, repeated).

Keeps full resolution and alternates a residual depthwise 9x9 conv (spatial mix)
with a pointwise 1x1 conv (channel mix), each with GroupNorm + SiLU. A simple,
strong all-conv design that rivals patch transformers. SiLU.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from model import NUM_FEATURES
from model_variants.heads import SpatialHead


class ConvMixerBlock(nn.Module):
    def __init__(self, dim: int, kernel: int = 9) -> None:
        super().__init__()
        self.dw = nn.Conv2d(dim, dim, kernel, padding=kernel // 2, groups=dim)
        self.norm1 = nn.GroupNorm(8, dim)
        self.pw = nn.Conv2d(dim, dim, kernel_size=1)
        self.norm2 = nn.GroupNorm(8, dim)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.norm1(self.act(self.dw(x)))  # residual depthwise
        return self.norm2(self.act(self.pw(x)))


class ConvMixer(nn.Module):
    def __init__(self, dim: int = 96, depth: int = 5, kernel: int = 9) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(NUM_FEATURES, dim, kernel_size=1), nn.SiLU(), nn.GroupNorm(8, dim)
        )
        self.blocks = nn.Sequential(
            *[ConvMixerBlock(dim, kernel) for _ in range(depth)]
        )
        self.head = SpatialHead(dim)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.head(self.blocks(self.stem(x)))


def build() -> ConvMixer:
    return ConvMixer()
