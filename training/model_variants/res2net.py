"""res2net — Res2Net multi-scale residual blocks.

Each block splits its channels into `scale` groups and processes them in a
hierarchical cascade: group i is convolved together with the (already
convolved) output of group i-1. This yields multiple effective receptive-field
sizes within a single block — multi-scale features at fine granularity. Width
128, full-res, SiLU.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from model import NUM_FEATURES
from model_variants.heads import SpatialHead


class Res2NetBlock(nn.Module):
    def __init__(self, channels: int, scale: int = 4) -> None:
        super().__init__()
        assert channels % scale == 0, "channels must be divisible by scale"
        self.scale = scale
        self.width = channels // scale
        self.norm = nn.GroupNorm(8, channels)
        self.act = nn.SiLU()
        self.convs = nn.ModuleList(
            [
                nn.Conv2d(self.width, self.width, kernel_size=3, padding=1)
                for _ in range(scale - 1)
            ]
        )
        self.proj = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.act(self.norm(x))
        spx = torch.split(h, self.width, dim=1)
        outs: list[torch.Tensor] = [spx[0]]
        sp = spx[0]
        for i in range(1, self.scale):
            inp = spx[i] if i == 1 else spx[i] + sp
            sp = self.act(self.convs[i - 1](inp))
            outs.append(sp)
        return x + self.proj(torch.cat(outs, dim=1))


class Res2Net(nn.Module):
    def __init__(self, hidden: int = 128, blocks: int = 4) -> None:
        super().__init__()
        self.stem = nn.Conv2d(NUM_FEATURES, hidden, kernel_size=3, padding=1)
        self.blocks = nn.Sequential(*[Res2NetBlock(hidden) for _ in range(blocks)])
        self.head = SpatialHead(hidden)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.head(self.blocks(self.stem(x)))


def build() -> Res2Net:
    return Res2Net()
