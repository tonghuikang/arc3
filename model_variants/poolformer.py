"""poolformer — MetaFormer with a parameter-free pooling token mixer.

PoolFormer's claim is that the *architecture* (norm -> token-mix -> norm ->
channel-MLP, both residual) matters more than the specific token mixer, so it
replaces attention with a plain average pool minus identity. Tests that thesis
on this task against the conv-heavy winners. Width 128, full-res, SiLU.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from model import NUM_FEATURES
from model_variants.heads import SpatialHead


class PoolFormerBlock(nn.Module):
    def __init__(self, channels: int, expansion: int = 4, pool: int = 3) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(1, channels)
        self.pool = nn.AvgPool2d(pool, stride=1, padding=pool // 2)
        self.norm2 = nn.GroupNorm(1, channels)
        self.pw1 = nn.Conv2d(channels, channels * expansion, kernel_size=1)
        self.act = nn.SiLU()
        self.pw2 = nn.Conv2d(channels * expansion, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n = self.norm1(x)
        x = x + (self.pool(n) - n)  # token mixer: local pooling minus identity
        x = x + self.pw2(self.act(self.pw1(self.norm2(x))))
        return x


class PoolFormer(nn.Module):
    def __init__(self, hidden: int = 128, blocks: int = 4) -> None:
        super().__init__()
        self.stem = nn.Conv2d(NUM_FEATURES, hidden, kernel_size=1)
        self.blocks = nn.Sequential(*[PoolFormerBlock(hidden) for _ in range(blocks)])
        self.head = SpatialHead(hidden)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.head(self.blocks(self.stem(x)))


def build() -> PoolFormer:
    return PoolFormer()
