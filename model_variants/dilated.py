"""dilated variant — a stack of dilated convs.

Exponentially growing dilation (1,2,4,8,16) gives a wide receptive field at full
64x64 resolution: a cell can integrate context from far across the board without
any downsampling or attention. A cheap middle ground between the per-pixel MLP
(no context) and the transformer (global, expensive).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from model import NUM_FEATURES
from model_variants.heads import SpatialHead


class DilatedCNN(nn.Module):
    def __init__(
        self, hidden: int = 96, dilations: tuple[int, ...] = (1, 2, 4, 8, 16)
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_c = NUM_FEATURES
        for d in dilations:
            layers.append(nn.Conv2d(in_c, hidden, kernel_size=3, padding=d, dilation=d))
            layers.append(nn.GELU())
            in_c = hidden
        self.body = nn.Sequential(*layers)
        self.head = SpatialHead(hidden)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.head(self.body(x))


def build() -> DilatedCNN:
    return DilatedCNN()
