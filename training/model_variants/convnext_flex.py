"""convnext_flex — shared flexible ConvNeXt base (NOT a variant: no build()).

The frontier-winning family is ConvNeXt and the width sweep located the sweet
spot near hidden=128. To probe the *other* block hyperparameters (depthwise
kernel size, pointwise expansion ratio) without touching the proven
``convnext.py``, thin experiment files import ``FlexConvNeXt`` from here and pin
the axis they vary. This module defines no ``build()``, so it is never itself
discovered as a variant (same convention as ``heads.py``).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from model import NUM_FEATURES
from model_variants.heads import SpatialHead


class FlexConvNeXtBlock(nn.Module):
    def __init__(self, channels: int, kernel: int = 7, expansion: int = 4) -> None:
        super().__init__()
        self.dw = nn.Conv2d(
            channels, channels, kernel_size=kernel, padding=kernel // 2, groups=channels
        )
        self.norm = nn.GroupNorm(1, channels)  # LayerNorm over channels
        self.pw1 = nn.Conv2d(channels, channels * expansion, kernel_size=1)
        self.act = nn.SiLU()
        self.pw2 = nn.Conv2d(channels * expansion, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.pw2(self.act(self.pw1(self.norm(self.dw(x)))))
        return x + h


class FlexConvNeXt(nn.Module):
    def __init__(
        self,
        hidden: int = 128,
        blocks: int = 4,
        kernel: int = 7,
        expansion: int = 4,
    ) -> None:
        super().__init__()
        self.stem = nn.Conv2d(NUM_FEATURES, hidden, kernel_size=1)
        self.blocks = nn.Sequential(
            *[FlexConvNeXtBlock(hidden, kernel, expansion) for _ in range(blocks)]
        )
        self.head = SpatialHead(hidden)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.head(self.blocks(self.stem(x)))
