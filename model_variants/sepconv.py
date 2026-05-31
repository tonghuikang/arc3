"""sepconv variant — depthwise-separable convolution stack.

Each layer is a depthwise 3x3 (spatial, per-channel) followed by a pointwise 1x1
(channel mixing) — the MobileNet factorisation. Cheap local spatial context with
far fewer params than dense 3x3 convs. Full-res, SiLU.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from model import NUM_FEATURES
from model_variants.heads import SpatialHead


class SepConv(nn.Module):
    def __init__(self, hidden: int = 96, layers: int = 5) -> None:
        super().__init__()
        mods: list[nn.Module] = []
        in_c = NUM_FEATURES
        for _ in range(layers):
            mods.append(nn.Conv2d(in_c, in_c, kernel_size=3, padding=1, groups=in_c))
            mods.append(nn.Conv2d(in_c, hidden, kernel_size=1))
            mods.append(nn.SiLU())
            in_c = hidden
        self.body = nn.Sequential(*mods)
        self.head = SpatialHead(hidden)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.head(self.body(x))


def build() -> SepConv:
    return SepConv()
