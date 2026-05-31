"""convnextv2 — ConvNeXt-V2 blocks with Global Response Normalization (GRN).

ConvNeXt-V2 adds GRN to the expanded MLP: it normalises each channel by its
global L2 response relative to the mean response, boosting feature diversity and
acting as a built-in regularizer (relevant given this task's overfitting). No
multiplicative gating on the main path. Width 128, full-res, SiLU.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from model import NUM_FEATURES
from model_variants.heads import SpatialHead


class GRN(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, dim, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gx = torch.norm(x, p=2, dim=(2, 3), keepdim=True)  # [B, C, 1, 1]
        nx = gx / (gx.mean(dim=1, keepdim=True) + 1e-6)
        return self.gamma * (x * nx) + self.beta + x


class ConvNeXtV2Block(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dw = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = nn.GroupNorm(1, dim)
        self.pw1 = nn.Conv2d(dim, dim * 4, kernel_size=1)
        self.act = nn.SiLU()
        self.grn = GRN(dim * 4)
        self.pw2 = nn.Conv2d(dim * 4, dim, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.grn(self.act(self.pw1(self.norm(self.dw(x)))))
        return x + self.pw2(h)


class ConvNeXtV2(nn.Module):
    def __init__(self, hidden: int = 128, blocks: int = 4) -> None:
        super().__init__()
        self.stem = nn.Conv2d(NUM_FEATURES, hidden, kernel_size=1)
        self.blocks = nn.Sequential(*[ConvNeXtV2Block(hidden) for _ in range(blocks)])
        self.head = SpatialHead(hidden)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.head(self.blocks(self.stem(x)))


def build() -> ConvNeXtV2:
    return ConvNeXtV2()
