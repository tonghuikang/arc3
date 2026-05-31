"""head_coordconv — convnext_h128 body + a CoordConv spatial click decoder.

Like head_clickconv but the click decoder is fed two extra channels holding the
normalised (x, y) coordinate of each cell. Plain convs are translation-invariant
and so are weak at *absolute* localization; CoordConv hands the head explicit
position, which is exactly what coordinate prediction needs. Body: convnext_h128.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from model import BOARD_SIZE, NUM_COLORS
from model_variants.cvx_body import ActionColorPool, ConvNeXtBody


class CoordConvClickHead(nn.Module):
    def __init__(self, in_ch: int, mid: int = 96, depth: int = 3) -> None:
        super().__init__()
        ys = torch.linspace(-1.0, 1.0, BOARD_SIZE)
        xs = torch.linspace(-1.0, 1.0, BOARD_SIZE)
        gy, gx = torch.meshgrid(ys, xs, indexing="ij")
        coords = torch.stack([gy, gx], dim=0).unsqueeze(0)  # [1, 2, 64, 64]
        self.register_buffer("coords", coords)
        layers: list[nn.Module] = []
        c = in_ch + 2
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
        b = fmap.shape[0]
        coords = self.coords.expand(b, -1, -1, -1)
        return self.click(torch.cat([fmap, coords], dim=1))


class Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.body = ConvNeXtBody(hidden=128, blocks=4)
        self.ac = ActionColorPool(self.body.out_channels)
        self.click = CoordConvClickHead(self.body.out_channels)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        f = self.body(x)
        action, color = self.ac(f)
        return action, color, self.click(f)


def build() -> Model:
    return Model()
