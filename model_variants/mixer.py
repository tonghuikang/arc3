"""mixer variant — MLP-Mixer over patch tokens (global mixing, no attention).

Alternates a *token-mixing* MLP (mixes across spatial positions) with a
*channel-mixing* MLP (mixes across features). It gets global receptive field
like a transformer but with plain MLPs instead of self-attention — a useful
point of comparison: how much of the transformer's value is attention per se vs.
just global mixing?
"""

from __future__ import annotations

import torch
import torch.nn as nn

from model import BOARD_SIZE, NUM_FEATURES
from model_variants.heads import SpatialHead


class MixerBlock(nn.Module):
    def __init__(self, n_tokens: int, dim: int) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.token_mlp = nn.Sequential(
            nn.Linear(n_tokens, n_tokens * 2),
            nn.GELU(),
            nn.Linear(n_tokens * 2, n_tokens),
        )
        self.norm2 = nn.LayerNorm(dim)
        self.channel_mlp = nn.Sequential(
            nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: [B, N, dim]
        y = self.norm1(x).transpose(1, 2)  # [B, dim, N]
        x = x + self.token_mlp(y).transpose(1, 2)
        x = x + self.channel_mlp(self.norm2(x))
        return x


class Mixer(nn.Module):
    def __init__(self, dim: int = 128, depth: int = 4, patch: int = 4) -> None:
        super().__init__()
        self.patch = patch
        self.grid = BOARD_SIZE // patch
        n_tokens = self.grid * self.grid
        self.embed = nn.Conv2d(NUM_FEATURES, dim, kernel_size=patch, stride=patch)
        self.blocks = nn.Sequential(*[MixerBlock(n_tokens, dim) for _ in range(depth)])
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
        b = x.shape[0]
        t = self.embed(x)  # [B, dim, grid, grid]
        g = t.shape[-1]
        t = t.flatten(2).transpose(1, 2)  # [B, N, dim]
        t = self.norm(self.blocks(t))
        fmap = t.transpose(1, 2).reshape(b, -1, g, g)  # [B, dim, grid, grid]
        return self.head(self.decode(fmap))  # upsample grid -> 64x64


def build() -> Mixer:
    return Mixer()
