"""vit variant — a spatial transformer (ViT-style) over the 64x64 board.

Unlike the per-coordinate MLP variants (``mlp.py``), which classify every pixel
independently with 1x1 convs and therefore cannot reason about relationships
*between* cells, this attends across the grid: it patch-embeds the board into
tokens, runs self-attention so any cell can condition on any other, then

- pools tokens for the global ``action``/``color`` heads, and
- decodes the per-token features back up to a full-resolution ``[B, 16, 64, 64]``
  click map (conv refine + learned upsample, so click precision isn't capped at
  patch resolution).

The shared :class:`SpatialTransformer` here is reused by the other ``vit_*``
variants. ``build()`` makes this file itself the default ViT.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from model import BOARD_SIZE, NUM_ACTIONS, NUM_COLORS, NUM_FEATURES


class SpatialTransformer(nn.Module):
    """Patch-embed -> transformer encoder -> (pooled action/color, dense click)."""

    def __init__(
        self,
        dim: int = 128,
        depth: int = 4,
        heads: int = 4,
        patch: int = 4,
        mlp_ratio: float = 4.0,
        conv_stem: int = 0,
        num_features: int = NUM_FEATURES,
    ) -> None:
        super().__init__()
        if BOARD_SIZE % patch != 0:
            raise ValueError(f"patch {patch} must divide board {BOARD_SIZE}")
        if dim % heads != 0:
            raise ValueError(f"dim {dim} must be divisible by heads {heads}")
        self.patch = patch
        self.grid = BOARD_SIZE // patch  # tokens per side
        n_tokens = self.grid * self.grid

        # Optional local conv stem (full-res 3x3 convs) before tokenisation — a
        # hybrid that mixes neighbourhood features locally, then globally via
        # attention. With conv_stem=0 this is a pure ViT.
        stem: list[nn.Module] = []
        in_c = num_features
        for _ in range(conv_stem):
            stem.append(nn.Conv2d(in_c, dim, kernel_size=3, padding=1))
            stem.append(nn.GELU())
            in_c = dim
        self.stem = nn.Sequential(*stem) if stem else nn.Identity()
        self.patch_embed = nn.Conv2d(in_c, dim, kernel_size=patch, stride=patch)
        self.pos = nn.Parameter(torch.zeros(1, n_tokens, dim))
        nn.init.trunc_normal_(self.pos, std=0.02)

        layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=int(dim * mlp_ratio),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=depth)
        self.norm = nn.LayerNorm(dim)

        self.action_head = nn.Linear(dim, NUM_ACTIONS)
        self.color_head = nn.Linear(dim, NUM_COLORS)
        # Decode token grid -> full-res per-colour click logits.
        self.click_decoder = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Upsample(scale_factor=patch, mode="nearest"),
            nn.Conv2d(dim, dim // 2, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(dim // 2, NUM_COLORS, kernel_size=1),
        )

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        b = x.shape[0]
        t = self.patch_embed(self.stem(x))  # [B, dim, grid, grid]
        g = t.shape[-1]
        t = t.flatten(2).transpose(1, 2)  # [B, N, dim]
        t = self.encoder(t + self.pos)
        t = self.norm(t)  # [B, N, dim]

        pooled = t.mean(dim=1)  # [B, dim]
        action_logits = self.action_head(pooled)  # [B, 6]
        color_logits = self.color_head(pooled)  # [B, 16]

        fmap = t.transpose(1, 2).reshape(b, -1, g, g)  # [B, dim, grid, grid]
        click_logits = self.click_decoder(fmap)  # [B, 16, 64, 64]
        return action_logits, color_logits, click_logits


def build() -> SpatialTransformer:
    return SpatialTransformer(dim=128, depth=4, heads=4, patch=4)
