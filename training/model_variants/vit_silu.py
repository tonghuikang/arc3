"""vit_silu experiment — the ViT architecture with SiLU activations throughout.

A controlled activation ablation against ``vit`` (GELU): identical
dim/depth/heads/patch, but SiLU in the transformer feed-forward and the click
decoder. Self-contained new code — the GELU ``vit.py`` is left untouched so its
results stay reproducible.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from model import BOARD_SIZE, NUM_ACTIONS, NUM_COLORS, NUM_FEATURES


class ViTSiLU(nn.Module):
    def __init__(
        self,
        dim: int = 128,
        depth: int = 4,
        heads: int = 4,
        patch: int = 4,
        mlp_ratio: float = 4.0,
    ) -> None:
        super().__init__()
        self.grid = BOARD_SIZE // patch
        n_tokens = self.grid * self.grid
        self.patch_embed = nn.Conv2d(NUM_FEATURES, dim, kernel_size=patch, stride=patch)
        self.pos = nn.Parameter(torch.zeros(1, n_tokens, dim))
        nn.init.trunc_normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=int(dim * mlp_ratio),
            activation=F.silu,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=depth)
        self.norm = nn.LayerNorm(dim)
        self.action_head = nn.Linear(dim, NUM_ACTIONS)
        self.color_head = nn.Linear(dim, NUM_COLORS)
        self.click_decoder = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Upsample(scale_factor=patch, mode="nearest"),
            nn.Conv2d(dim, dim // 2, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(dim // 2, NUM_COLORS, kernel_size=1),
        )

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        b = x.shape[0]
        t = self.patch_embed(x)
        g = t.shape[-1]
        t = t.flatten(2).transpose(1, 2)
        t = self.norm(self.encoder(t + self.pos))
        pooled = t.mean(dim=1)
        fmap = t.transpose(1, 2).reshape(b, -1, g, g)
        return (
            self.action_head(pooled),
            self.color_head(pooled),
            self.click_decoder(fmap),
        )


def build() -> ViTSiLU:
    return ViTSiLU()
