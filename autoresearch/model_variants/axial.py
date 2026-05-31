"""axial variant — axial self-attention (attend along rows, then columns).

Full 2-D self-attention over 32x32 tokens would be a 1024x1024 attention matrix;
axial attention factorises it into a row pass (each cell attends within its row)
and a column pass (within its column), reaching every other cell in two steps at
a fraction of the cost. Keeps a finer token grid (patch 2) than the patch-4 ViT.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from model import BOARD_SIZE, NUM_FEATURES
from model_variants.heads import SpatialHead


class AxialBlock(nn.Module):
    def __init__(self, dim: int, heads: int) -> None:
        super().__init__()
        self.norm_r = nn.LayerNorm(dim)
        self.row_attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm_c = nn.LayerNorm(dim)
        self.col_attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm_m = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: [B, H, W, dim]
        b, h, w, c = x.shape
        r = self.norm_r(x).reshape(b * h, w, c)  # rows as sequences
        x = x + self.row_attn(r, r, r, need_weights=False)[0].reshape(b, h, w, c)
        col = self.norm_c(x).transpose(1, 2).reshape(b * w, h, c)  # cols as sequences
        attn = self.col_attn(col, col, col, need_weights=False)[0]
        x = x + attn.reshape(b, w, h, c).transpose(1, 2)
        x = x + self.mlp(self.norm_m(x))
        return x


class AxialTransformer(nn.Module):
    def __init__(
        self, dim: int = 64, depth: int = 4, heads: int = 4, patch: int = 2
    ) -> None:
        super().__init__()
        if dim % heads != 0:
            raise ValueError(f"dim {dim} must be divisible by heads {heads}")
        self.patch = patch
        self.grid = BOARD_SIZE // patch
        self.embed = nn.Conv2d(NUM_FEATURES, dim, kernel_size=patch, stride=patch)
        self.row_pos = nn.Parameter(torch.zeros(1, self.grid, 1, dim))
        self.col_pos = nn.Parameter(torch.zeros(1, 1, self.grid, dim))
        nn.init.trunc_normal_(self.row_pos, std=0.02)
        nn.init.trunc_normal_(self.col_pos, std=0.02)
        self.blocks = nn.ModuleList([AxialBlock(dim, heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.decode = nn.Sequential(
            nn.Upsample(scale_factor=patch, mode="nearest"),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.head = SpatialHead(dim)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        t = self.embed(x)  # [B, dim, grid, grid]
        t = t.permute(0, 2, 3, 1)  # [B, H, W, dim]
        t = t + self.row_pos + self.col_pos
        for block in self.blocks:
            t = block(t)
        t = self.norm(t).permute(0, 3, 1, 2)  # [B, dim, grid, grid]
        return self.head(self.decode(t))


def build() -> AxialTransformer:
    return AxialTransformer()
