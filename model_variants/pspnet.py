"""pspnet variant — pyramid pooling module (multi-scale global context).

Pools the conv feature map to several grid sizes (1x1, 2x2, 4x4, 8x8), processes
each, upsamples back, and concatenates with the full-res features. Each branch
injects context at a different scale — a non-attention way to see globally.
Full-res output, SiLU.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from model import NUM_FEATURES
from model_variants.heads import SpatialHead


class PSPNet(nn.Module):
    def __init__(
        self, hidden: int = 96, scales: tuple[int, ...] = (1, 2, 4, 8)
    ) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(NUM_FEATURES, hidden, kernel_size=3, padding=1), nn.SiLU()
        )
        branch_c = hidden // 4
        self.branches = nn.ModuleList(
            [
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(s),
                    nn.Conv2d(hidden, branch_c, kernel_size=1),
                    nn.SiLU(),
                )
                for s in scales
            ]
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(
                hidden + branch_c * len(scales), hidden, kernel_size=3, padding=1
            ),
            nn.SiLU(),
        )
        self.head = SpatialHead(hidden)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.stem(x)
        size = h.shape[-2:]
        feats = [h]
        for branch in self.branches:
            p = branch(h)
            feats.append(
                F.interpolate(p, size=size, mode="bilinear", align_corners=False)
            )
        return self.head(self.fuse(torch.cat(feats, dim=1)))


def build() -> PSPNet:
    return PSPNet()
