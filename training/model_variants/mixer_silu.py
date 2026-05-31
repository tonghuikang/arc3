"""mixer_silu experiment — the MLP-Mixer backbone with SiLU activations.

Activation ablation against ``mixer`` (GELU): same token-/channel-mixing MLPs
and patch grid, SiLU instead of GELU. Self-contained new code.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from model import BOARD_SIZE, NUM_FEATURES
from model_variants.heads import SpatialHead


class MixerBlockSiLU(nn.Module):
    def __init__(self, n_tokens: int, dim: int) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.token_mlp = nn.Sequential(
            nn.Linear(n_tokens, n_tokens * 2),
            nn.SiLU(),
            nn.Linear(n_tokens * 2, n_tokens),
        )
        self.norm2 = nn.LayerNorm(dim)
        self.channel_mlp = nn.Sequential(
            nn.Linear(dim, dim * 4), nn.SiLU(), nn.Linear(dim * 4, dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.norm1(x).transpose(1, 2)
        x = x + self.token_mlp(y).transpose(1, 2)
        x = x + self.channel_mlp(self.norm2(x))
        return x


class MixerSiLU(nn.Module):
    def __init__(self, dim: int = 128, depth: int = 4, patch: int = 4) -> None:
        super().__init__()
        self.grid = BOARD_SIZE // patch
        n_tokens = self.grid * self.grid
        self.embed = nn.Conv2d(NUM_FEATURES, dim, kernel_size=patch, stride=patch)
        self.blocks = nn.Sequential(
            *[MixerBlockSiLU(n_tokens, dim) for _ in range(depth)]
        )
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


def build() -> MixerSiLU:
    return MixerSiLU()
