"""coordconv variant — per-pixel convs with appended x/y coordinate channels.

CoordConv: concatenates two normalised coordinate planes (x in [-1,1], y in
[-1,1]) to the input so the network has absolute position awareness, then 3x3
convs for local context. Tests whether telling the (otherwise position-blind)
conv where each cell is helps. SiLU activations.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from model import NUM_FEATURES
from model_variants.heads import SpatialHead


class CoordConv(nn.Module):
    def __init__(self, hidden: int = 64, layers: int = 3) -> None:
        super().__init__()
        convs: list[nn.Module] = []
        in_c = NUM_FEATURES + 2  # + x, y coordinate planes
        for _ in range(layers):
            convs.append(nn.Conv2d(in_c, hidden, kernel_size=3, padding=1))
            convs.append(nn.SiLU())
            in_c = hidden
        self.body = nn.Sequential(*convs)
        self.head = SpatialHead(hidden)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        b, _, h, w = x.shape
        ys = (
            torch.linspace(-1, 1, h, device=x.device)
            .view(1, 1, h, 1)
            .expand(b, 1, h, w)
        )
        xs = (
            torch.linspace(-1, 1, w, device=x.device)
            .view(1, 1, 1, w)
            .expand(b, 1, h, w)
        )
        return self.head(self.body(torch.cat([x, xs, ys], dim=1)))


def build() -> CoordConv:
    return CoordConv()
