"""gmlp variant — gated MLP with a Spatial Gating Unit (no attention).

gMLP: per block, project up, then a Spatial Gating Unit splits channels and gates
one half by a learned *spatial* (token-mixing) linear map of the other, then
project down. Captures spatial interactions without attention. Patch 4 tokens,
SiLU.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from model import BOARD_SIZE, NUM_FEATURES
from model_variants.heads import SpatialHead


class SpatialGatingUnit(nn.Module):
    def __init__(self, dim: int, n_tokens: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim // 2)
        self.proj = nn.Linear(n_tokens, n_tokens)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # [B, N, dim]
        u, v = x.chunk(2, dim=-1)
        v = self.norm(v).transpose(1, 2)  # [B, dim/2, N]
        v = self.proj(v).transpose(1, 2)  # spatial (token) mixing
        return u * v


class GMLPBlock(nn.Module):
    def __init__(self, dim: int, n_tokens: int, ffn: int = 4) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.proj_in = nn.Linear(dim, dim * ffn)
        self.act = nn.SiLU()
        self.sgu = SpatialGatingUnit(dim * ffn, n_tokens)
        self.proj_out = nn.Linear(dim * ffn // 2, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.act(self.proj_in(self.norm(x)))
        return x + self.proj_out(self.sgu(h))


class GMLP(nn.Module):
    def __init__(self, dim: int = 128, depth: int = 4, patch: int = 4) -> None:
        super().__init__()
        self.grid = BOARD_SIZE // patch
        n_tokens = self.grid * self.grid
        self.embed = nn.Conv2d(NUM_FEATURES, dim, kernel_size=patch, stride=patch)
        self.blocks = nn.Sequential(*[GMLPBlock(dim, n_tokens) for _ in range(depth)])
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
        b = x.shape[0]
        t = self.embed(x)
        g = t.shape[-1]
        t = t.flatten(2).transpose(1, 2)
        t = self.norm(self.blocks(t))
        fmap = t.transpose(1, 2).reshape(b, -1, g, g)
        return self.head(self.decode(fmap))


def build() -> GMLP:
    return GMLP()
