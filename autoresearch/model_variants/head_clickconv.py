"""head_clickconv — convnext_h128 body + a spatial click DECODER.

The shared SpatialHead predicts each click cell from that cell's feature vector
alone (a single 1x1 conv = zero head receptive field). Localization needs to
compare neighbouring cells, so here the click map comes from a small stack of
3x3 convs. Action/color use the proven amax-pool head, unchanged — a clean A/B
on the dominant coord term. Body: convnext_h128.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from model import NUM_COLORS
from model_variants.cvx_body import ActionColorPool, ConvNeXtBody


class ConvClickHead(nn.Module):
    def __init__(self, in_ch: int, mid: int = 96, depth: int = 3) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        c = in_ch
        for _ in range(depth):
            layers += [
                nn.Conv2d(c, mid, kernel_size=3, padding=1),
                nn.GroupNorm(8, mid),
                nn.SiLU(),
            ]
            c = mid
        layers += [nn.Conv2d(mid, NUM_COLORS, kernel_size=1)]
        self.click = nn.Sequential(*layers)

    def forward(self, fmap: torch.Tensor) -> torch.Tensor:
        return self.click(fmap)


class Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.body = ConvNeXtBody(hidden=128, blocks=4)
        self.ac = ActionColorPool(self.body.out_channels)
        self.click = ConvClickHead(self.body.out_channels)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        f = self.body(x)
        action, color = self.ac(f)
        return action, color, self.click(f)


def build() -> Model:
    return Model()
