"""eca_convnext — ConvNeXt blocks with Efficient Channel Attention (ECA).

Like se_convnext but the channel-attention gate is parameter-light: instead of a
two-layer squeeze MLP, ECA applies a single 1-D conv across the globally-pooled
channel descriptor (local cross-channel interaction, no dimensionality
reduction). Width 128, full-res, SiLU.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from model import NUM_FEATURES
from model_variants.heads import SpatialHead


class ECAGate(nn.Module):
    def __init__(self, kernel_size: int = 5) -> None:
        super().__init__()
        self.conv = nn.Conv1d(
            1, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = x.mean(dim=(2, 3))  # [B, C]
        s = self.conv(s.unsqueeze(1))  # [B, 1, C]
        s = torch.sigmoid(s).squeeze(1)  # [B, C]
        return x * s.unsqueeze(-1).unsqueeze(-1)


class ECAConvNeXtBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.dw = nn.Conv2d(
            channels, channels, kernel_size=7, padding=3, groups=channels
        )
        self.norm = nn.GroupNorm(1, channels)
        self.pw1 = nn.Conv2d(channels, channels * 4, kernel_size=1)
        self.act = nn.SiLU()
        self.pw2 = nn.Conv2d(channels * 4, channels, kernel_size=1)
        self.eca = ECAGate()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.pw2(self.act(self.pw1(self.norm(self.dw(x)))))
        return x + self.eca(h)


class ECAConvNeXt(nn.Module):
    def __init__(self, hidden: int = 128, blocks: int = 4) -> None:
        super().__init__()
        self.stem = nn.Conv2d(NUM_FEATURES, hidden, kernel_size=1)
        self.blocks = nn.Sequential(*[ECAConvNeXtBlock(hidden) for _ in range(blocks)])
        self.head = SpatialHead(hidden)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.head(self.blocks(self.stem(x)))


def build() -> ECAConvNeXt:
    return ECAConvNeXt()
