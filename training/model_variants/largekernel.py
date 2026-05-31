"""largekernel variant — a stack of large 7x7 convolutions.

Plainest way to get a wide receptive field: full 7x7 convs (no dilation, no
downsampling). Each layer sees a 7x7 neighbourhood directly; four layers reach
~25x25. A simple-but-heavy contrast to the dilated/attention routes. SiLU.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from model import NUM_FEATURES
from model_variants.heads import SpatialHead


class LargeKernelCNN(nn.Module):
    def __init__(self, hidden: int = 64, layers: int = 4, kernel: int = 7) -> None:
        super().__init__()
        convs: list[nn.Module] = []
        in_c = NUM_FEATURES
        for _ in range(layers):
            convs.append(
                nn.Conv2d(in_c, hidden, kernel_size=kernel, padding=kernel // 2)
            )
            convs.append(nn.SiLU())
            in_c = hidden
        self.body = nn.Sequential(*convs)
        self.head = SpatialHead(hidden)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.head(self.body(x))


def build() -> LargeKernelCNN:
    return LargeKernelCNN()
