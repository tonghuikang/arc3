"""fnet variant — FNet-style Fourier token mixing (global mixing, no attention).

Replaces self-attention with a parameter-free 2-D FFT (real part) over the patch
tokens, followed by a SiLU MLP. Mixes information globally far more cheaply than
attention — a useful baseline for "how much does learned attention matter vs.
just any global mixing?".
"""

from __future__ import annotations

import torch
import torch.nn as nn

from model import BOARD_SIZE, NUM_FEATURES
from model_variants.heads import SpatialHead


class FNetBlock(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4), nn.SiLU(), nn.Linear(dim * 4, dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # [B, N, dim]
        mixed = torch.fft.fft2(x.to(torch.float32)).real  # mix tokens + features
        x = self.norm1(x + mixed)
        x = self.norm2(x + self.mlp(x))
        return x


class FNet(nn.Module):
    def __init__(self, dim: int = 128, depth: int = 4, patch: int = 4) -> None:
        super().__init__()
        self.grid = BOARD_SIZE // patch
        self.embed = nn.Conv2d(NUM_FEATURES, dim, kernel_size=patch, stride=patch)
        self.blocks = nn.Sequential(*[FNetBlock(dim) for _ in range(depth)])
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


def build() -> FNet:
    return FNet()
