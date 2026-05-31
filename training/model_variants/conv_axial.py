"""conv_axial variant — conv stem then axial attention (local-then-global hybrid).

Two full-res 3x3 convs mix neighbourhoods locally, then patch-2 tokens are mixed
globally by axial (row+column) attention. Combines the conv family's local
precision with attention's long-range reach. Reuses the SiLU axial block. SiLU.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from model import BOARD_SIZE, NUM_FEATURES
from model_variants.axial_silu import AxialBlockSiLU
from model_variants.heads import SpatialHead


class ConvAxial(nn.Module):
    def __init__(
        self,
        dim: int = 64,
        depth: int = 4,
        heads: int = 4,
        patch: int = 2,
        stem: int = 2,
    ) -> None:
        super().__init__()
        convs: list[nn.Module] = []
        in_c = NUM_FEATURES
        for _ in range(stem):
            convs.append(nn.Conv2d(in_c, dim, kernel_size=3, padding=1))
            convs.append(nn.SiLU())
            in_c = dim
        self.stem = nn.Sequential(*convs)
        self.grid = BOARD_SIZE // patch
        self.embed = nn.Conv2d(dim, dim, kernel_size=patch, stride=patch)
        self.row_pos = nn.Parameter(torch.zeros(1, self.grid, 1, dim))
        self.col_pos = nn.Parameter(torch.zeros(1, 1, self.grid, dim))
        nn.init.trunc_normal_(self.row_pos, std=0.02)
        nn.init.trunc_normal_(self.col_pos, std=0.02)
        self.blocks = nn.ModuleList([AxialBlockSiLU(dim, heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.decode = nn.Sequential(
            nn.Upsample(scale_factor=patch, mode="nearest"),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1),
            nn.SiLU(),
        )
        self.head = SpatialHead(dim)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        t = self.embed(self.stem(x)).permute(0, 2, 3, 1)
        t = t + self.row_pos + self.col_pos
        for block in self.blocks:
            t = block(t)
        t = self.norm(t).permute(0, 3, 1, 2)
        return self.head(self.decode(t))


def build() -> ConvAxial:
    return ConvAxial()
