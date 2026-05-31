"""aspp variant — Atrous Spatial Pyramid Pooling (parallel dilations + global).

Runs parallel atrous convs at dilations 1/6/12/18 plus an image-level global-pool
branch over the same features, then fuses. Captures multiple receptive-field
scales and global context in a single layer. Full-res, SiLU.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from model import NUM_FEATURES
from model_variants.heads import SpatialHead


class ASPP(nn.Module):
    def __init__(
        self, hidden: int = 96, dilations: tuple[int, ...] = (1, 6, 12, 18)
    ) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(NUM_FEATURES, hidden, kernel_size=1), nn.SiLU()
        )
        self.branches = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(hidden, hidden, kernel_size=3, padding=d, dilation=d),
                    nn.SiLU(),
                )
                for d in dilations
            ]
        )
        self.gpool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Conv2d(hidden, hidden, kernel_size=1), nn.SiLU()
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(hidden * (len(dilations) + 1), hidden, kernel_size=1), nn.SiLU()
        )
        self.head = SpatialHead(hidden)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.stem(x)
        size = h.shape[-2:]
        feats = [branch(h) for branch in self.branches]
        g = F.interpolate(
            self.gpool(h), size=size, mode="bilinear", align_corners=False
        )
        feats.append(g)
        return self.head(self.fuse(torch.cat(feats, dim=1)))


def build() -> ASPP:
    return ASPP()
