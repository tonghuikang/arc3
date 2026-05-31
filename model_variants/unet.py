"""unet variant — multi-scale conv encoder/decoder with skip connections.

Downsamples 64 -> 32 -> 16 (growing channels), then upsamples back to 64,
concatenating the matching encoder feature at each scale. The bottleneck mixes
information globally at coarse resolution while the skip connections preserve
fine click-precision detail — a different way to get long-range context than
attention.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from model import NUM_FEATURES
from model_variants.heads import SpatialHead


def _cbr(in_c: int, out_c: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
        nn.GELU(),
        nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
        nn.GELU(),
    )


class UNet(nn.Module):
    def __init__(self, base: int = 48) -> None:
        super().__init__()
        self.enc1 = _cbr(NUM_FEATURES, base)  # 64x64
        self.enc2 = _cbr(base, base * 2)  # 32x32
        self.bott = _cbr(base * 2, base * 4)  # 16x16
        self.pool = nn.MaxPool2d(2)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, kernel_size=2, stride=2)
        self.dec2 = _cbr(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, kernel_size=2, stride=2)
        self.dec1 = _cbr(base * 2, base)
        self.head = SpatialHead(base)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        e1 = self.enc1(x)  # [B, base, 64, 64]
        e2 = self.enc2(self.pool(e1))  # [B, 2base, 32, 32]
        b = self.bott(self.pool(e2))  # [B, 4base, 16, 16]
        d2 = self.dec2(torch.cat([self.up2(b), e2], dim=1))  # [B, 2base, 32, 32]
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))  # [B, base, 64, 64]
        return self.head(d1)


def build() -> UNet:
    return UNet()
