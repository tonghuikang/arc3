"""head_combo — convnext_h128 body + the best-guess head on every axis.

Kitchen-sink candidate that stacks the individually-motivated improvements:
attention pooling for action/color, and a CoordConv spatial click decoder whose
final map is the outer sum of x/y marginals (separable prior) plus a full 2-D
residual. If the isolated head experiments each help, this is the candidate that
should clear the convnext_h128 plateau. Body: convnext_h128.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from model import BOARD_SIZE, NUM_ACTIONS, NUM_COLORS
from model_variants.cvx_body import ConvNeXtBody


class ComboHead(nn.Module):
    def __init__(self, in_ch: int, mid: int = 96, depth: int = 3) -> None:
        super().__init__()
        n = NUM_ACTIONS + NUM_COLORS
        self.val = nn.Conv2d(in_ch, n, kernel_size=1)
        self.att = nn.Conv2d(in_ch, n, kernel_size=1)
        # CoordConv click decoder
        ys = torch.linspace(-1.0, 1.0, BOARD_SIZE)
        xs = torch.linspace(-1.0, 1.0, BOARD_SIZE)
        gy, gx = torch.meshgrid(ys, xs, indexing="ij")
        self.register_buffer("coords", torch.stack([gy, gx], 0).unsqueeze(0))
        layers: list[nn.Module] = []
        c = in_ch + 2
        for _ in range(depth):
            layers += [
                nn.Conv2d(c, mid, kernel_size=3, padding=1),
                nn.GroupNorm(8, mid),
                nn.SiLU(),
            ]
            c = mid
        self.click_feat = nn.Sequential(*layers)
        self.click_2d = nn.Conv2d(mid, NUM_COLORS, kernel_size=1)  # full 2-D map
        self.click_row = nn.Conv2d(mid, NUM_COLORS, kernel_size=1)  # y marginal
        self.click_col = nn.Conv2d(mid, NUM_COLORS, kernel_size=1)  # x marginal

    def forward(
        self, fmap: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # attention-pooled action/color
        att = self.att(fmap)
        w = torch.softmax(att.flatten(2), dim=2).view_as(att)
        pooled = (self.val(fmap) * w).sum(dim=(2, 3))
        action = pooled[:, :NUM_ACTIONS]
        color = pooled[:, NUM_ACTIONS:]
        # separable + 2-D residual click map
        b = fmap.shape[0]
        cf = self.click_feat(torch.cat([fmap, self.coords.expand(b, -1, -1, -1)], 1))
        row = self.click_row(cf).amax(dim=3)  # [B,16,H]
        col = self.click_col(cf).amax(dim=2)  # [B,16,W]
        click = row.unsqueeze(3) + col.unsqueeze(2) + self.click_2d(cf)
        return action, color, click


class Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.body = ConvNeXtBody(hidden=128, blocks=4)
        self.head = ComboHead(self.body.out_channels)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.head(self.body(x))


def build() -> Model:
    return Model()
