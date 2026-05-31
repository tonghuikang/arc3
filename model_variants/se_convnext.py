"""se_convnext — ConvNeXt blocks augmented with Squeeze-and-Excitation gating.

Combines the two strongest design points on the current frontier: ConvNeXt's
7x7-depthwise + pointwise-MLP block and SE channel attention. The SE gate
rescales channels using global context just before each residual add. Width 128
(the winning width), full-res, SiLU throughout.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from model import NUM_FEATURES
from model_variants.heads import SpatialHead


class SEGate(nn.Module):
    def __init__(self, channels: int, reduction: int = 8) -> None:
        super().__init__()
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.fc2 = nn.Linear(channels // reduction, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = F.silu(self.fc1(x.mean(dim=(2, 3))))
        s = torch.sigmoid(self.fc2(s))
        return x * s.unsqueeze(-1).unsqueeze(-1)


class SEConvNeXtBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.dw = nn.Conv2d(
            channels, channels, kernel_size=7, padding=3, groups=channels
        )
        self.norm = nn.GroupNorm(1, channels)
        self.pw1 = nn.Conv2d(channels, channels * 4, kernel_size=1)
        self.act = nn.SiLU()
        self.pw2 = nn.Conv2d(channels * 4, channels, kernel_size=1)
        self.se = SEGate(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.pw2(self.act(self.pw1(self.norm(self.dw(x)))))
        return x + self.se(h)


class SEConvNeXt(nn.Module):
    def __init__(self, hidden: int = 128, blocks: int = 4) -> None:
        super().__init__()
        self.stem = nn.Conv2d(NUM_FEATURES, hidden, kernel_size=1)
        self.blocks = nn.Sequential(*[SEConvNeXtBlock(hidden) for _ in range(blocks)])
        self.head = SpatialHead(hidden)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.head(self.blocks(self.stem(x)))


def build() -> SEConvNeXt:
    return SEConvNeXt()
