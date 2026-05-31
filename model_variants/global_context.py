"""global_context variant — per-pixel conv plus a broadcast global context vector.

Directly targets the per-pixel models' blind spot (they can't see globally): a
1x1 conv extracts local features, a global average pool + MLP summarises the
whole board, and that summary is broadcast back and concatenated to every cell
before the output head. Cheap way to give a per-pixel model global awareness.
SiLU activations.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from model import NUM_FEATURES
from model_variants.heads import SpatialHead


class GlobalContext(nn.Module):
    def __init__(self, hidden: int = 96) -> None:
        super().__init__()
        self.local = nn.Sequential(
            nn.Conv2d(NUM_FEATURES, hidden, kernel_size=1), nn.SiLU()
        )
        self.ctx = nn.Sequential(
            nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, hidden)
        )
        self.mix = nn.Sequential(
            nn.Conv2d(hidden * 2, hidden, kernel_size=1), nn.SiLU()
        )
        self.head = SpatialHead(hidden)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.local(x)  # [B, hidden, 64, 64]
        g = self.ctx(h.mean(dim=(2, 3)))  # [B, hidden] global summary
        g = g.unsqueeze(-1).unsqueeze(-1).expand_as(h)
        return self.head(self.mix(torch.cat([h, g], dim=1)))


def build() -> GlobalContext:
    return GlobalContext()
