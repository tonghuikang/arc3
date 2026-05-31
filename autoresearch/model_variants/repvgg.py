"""repvgg variant — RepVGG-style multi-branch conv blocks.

Each block sums a 3x3 conv, a 1x1 conv, and (when shapes match) an identity
branch before the activation — multiple parallel paths that enrich training-time
gradients. Plain, full-res, SiLU.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from model import NUM_FEATURES
from model_variants.heads import SpatialHead


class RepVGGBlock(nn.Module):
    def __init__(self, in_c: int, out_c: int) -> None:
        super().__init__()
        self.conv3 = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1)
        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=1)
        self.has_identity = in_c == out_c
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv3(x) + self.conv1(x)
        if self.has_identity:
            y = y + x
        return self.act(y)


class RepVGG(nn.Module):
    def __init__(self, hidden: int = 96, layers: int = 5) -> None:
        super().__init__()
        blocks: list[nn.Module] = []
        in_c = NUM_FEATURES
        for _ in range(layers):
            blocks.append(RepVGGBlock(in_c, hidden))
            in_c = hidden
        self.body = nn.Sequential(*blocks)
        self.head = SpatialHead(hidden)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.head(self.body(x))


def build() -> RepVGG:
    return RepVGG()
