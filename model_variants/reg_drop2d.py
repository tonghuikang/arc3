"""reg_drop2d — convnext_h128 with spatial dropout between blocks (regularizer).

The sweep's limit is overfitting: the body memorizes training click/colour
targets (train coord ~0.005) but the held-out base games have different targets
(val coord ~1.86). Spatial dropout (Dropout2d zeroes whole channels) between
ConvNeXt blocks forces redundant, less memorised features. Body: convnext_h128,
p=0.2.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from model import NUM_FEATURES
from model_variants.convnext_flex import FlexConvNeXtBlock
from model_variants.heads import SpatialHead


class RegConvNeXt(nn.Module):
    def __init__(self, hidden: int = 128, blocks: int = 4, p: float = 0.2) -> None:
        super().__init__()
        self.stem = nn.Conv2d(NUM_FEATURES, hidden, kernel_size=1)
        self.blocks = nn.ModuleList([FlexConvNeXtBlock(hidden) for _ in range(blocks)])
        self.drop = nn.Dropout2d(p)
        self.head = SpatialHead(hidden)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.stem(x)
        for blk in self.blocks:
            h = self.drop(blk(h))
        return self.head(h)


def build() -> RegConvNeXt:
    return RegConvNeXt()
