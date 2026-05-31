"""mobilenetv2 — inverted-residual (MBConv) blocks.

The MobileNetV2 block: 1x1 expand -> 3x3 depthwise -> 1x1 project, with the
residual on the *narrow* (un-expanded) ends. Cheap, depthwise-separable spatial
mixing — a different efficiency point from ConvNeXt's depthwise-first design.
Width 128, expansion 4, full-res, SiLU.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from model import NUM_FEATURES
from model_variants.heads import SpatialHead


class InvertedResidual(nn.Module):
    def __init__(self, channels: int, expansion: int = 4) -> None:
        super().__init__()
        mid = channels * expansion
        self.norm = nn.GroupNorm(8, channels)
        self.expand = nn.Conv2d(channels, mid, kernel_size=1)
        self.dw = nn.Conv2d(mid, mid, kernel_size=3, padding=1, groups=mid)
        self.project = nn.Conv2d(mid, channels, kernel_size=1)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        h = self.act(self.expand(h))
        h = self.act(self.dw(h))
        h = self.project(h)
        return x + h


class MobileNetV2(nn.Module):
    def __init__(self, hidden: int = 128, blocks: int = 4) -> None:
        super().__init__()
        self.stem = nn.Conv2d(NUM_FEATURES, hidden, kernel_size=3, padding=1)
        self.blocks = nn.Sequential(*[InvertedResidual(hidden) for _ in range(blocks)])
        self.head = SpatialHead(hidden)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.head(self.blocks(self.stem(x)))


def build() -> MobileNetV2:
    return MobileNetV2()
