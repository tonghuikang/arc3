"""resnet_silu experiment — the residual conv backbone with SiLU activations.

Activation ablation against ``resnet`` (GELU): same depth/width and GroupNorm,
SiLU instead of GELU. Self-contained new code.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from model import NUM_FEATURES
from model_variants.heads import SpatialHead


class ResBlockSiLU(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(8, channels)
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(8, channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = self.conv2(F.silu(self.norm2(h)))
        return x + h


class ResNetPolicySiLU(nn.Module):
    def __init__(self, hidden: int = 96, blocks: int = 5) -> None:
        super().__init__()
        self.stem = nn.Conv2d(NUM_FEATURES, hidden, kernel_size=3, padding=1)
        self.blocks = nn.Sequential(*[ResBlockSiLU(hidden) for _ in range(blocks)])
        self.head = SpatialHead(hidden)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.head(self.blocks(self.stem(x)))


def build() -> ResNetPolicySiLU:
    return ResNetPolicySiLU()
