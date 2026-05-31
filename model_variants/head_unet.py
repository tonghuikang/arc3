"""head_unet — convnext_h128 body + a U-Net click decoder.

Gives the click head multi-scale context: downsample to 32x32 and 16x16, then
upsample back with skip connections. Coarse levels see large neighbourhoods
(where roughly to click); the full-res skip keeps the peak crisp. Action/color
use the proven amax-pool head. Body: convnext_h128.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from model import NUM_COLORS
from model_variants.cvx_body import ActionColorPool, ConvNeXtBody


def _cbr(c_in: int, c_out: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, kernel_size=3, padding=1),
        nn.GroupNorm(8, c_out),
        nn.SiLU(),
    )


class UNetClickHead(nn.Module):
    def __init__(self, in_ch: int, c: int = 64) -> None:
        super().__init__()
        self.enc0 = _cbr(in_ch, c)  # 64x64
        self.down1 = nn.MaxPool2d(2)
        self.enc1 = _cbr(c, c * 2)  # 32x32
        self.down2 = nn.MaxPool2d(2)
        self.enc2 = _cbr(c * 2, c * 2)  # 16x16
        self.up1 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec1 = _cbr(c * 2 + c * 2, c * 2)  # 32x32 (+ skip enc1)
        self.up0 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec0 = _cbr(c * 2 + c, c)  # 64x64 (+ skip enc0)
        self.out = nn.Conv2d(c, NUM_COLORS, kernel_size=1)

    def forward(self, fmap: torch.Tensor) -> torch.Tensor:
        e0 = self.enc0(fmap)
        e1 = self.enc1(self.down1(e0))
        e2 = self.enc2(self.down2(e1))
        d1 = self.dec1(torch.cat([self.up1(e2), e1], dim=1))
        d0 = self.dec0(torch.cat([self.up0(d1), e0], dim=1))
        return self.out(d0)


class Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.body = ConvNeXtBody(hidden=128, blocks=4)
        self.ac = ActionColorPool(self.body.out_channels)
        self.click = UNetClickHead(self.body.out_channels)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        f = self.body(x)
        action, color = self.ac(f)
        return action, color, self.click(f)


def build() -> Model:
    return Model()
