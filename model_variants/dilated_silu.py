"""dilated_silu experiment — the dilated-conv backbone with SiLU activations.

Activation ablation against ``dilated`` (GELU): same dilation schedule and
width, SiLU instead of GELU. Self-contained new code.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from model import NUM_FEATURES
from model_variants.heads import SpatialHead


class DilatedCNNSiLU(nn.Module):
    def __init__(
        self, hidden: int = 96, dilations: tuple[int, ...] = (1, 2, 4, 8, 16)
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_c = NUM_FEATURES
        for d in dilations:
            layers.append(nn.Conv2d(in_c, hidden, kernel_size=3, padding=d, dilation=d))
            layers.append(nn.SiLU())
            in_c = hidden
        self.body = nn.Sequential(*layers)
        self.head = SpatialHead(hidden)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.head(self.body(x))


def build() -> DilatedCNNSiLU:
    return DilatedCNNSiLU()
