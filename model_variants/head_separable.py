"""head_separable — convnext_h128 body + a factorized x/y (separable) click head.

Imposes a strong localization prior: the per-colour click map is the outer sum
of a row-logit over y and a column-logit over x, i.e. log P(click) =
log P(y) + log P(x). A click is thus predicted by two 64-way 1-D decisions
instead of one 4096-way 2-D one — far fewer effective parameters to localise,
which often sharpens coordinate prediction. Output stays [B, 16, 64, 64].
Body: convnext_h128.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from model import NUM_COLORS
from model_variants.cvx_body import ActionColorPool, ConvNeXtBody


class SeparableClickHead(nn.Module):
    def __init__(self, in_ch: int) -> None:
        super().__init__()
        # Per-colour evidence maps; collapse one axis to get each marginal.
        self.row = nn.Conv2d(in_ch, NUM_COLORS, kernel_size=1)  # -> pool over x (W)
        self.col = nn.Conv2d(in_ch, NUM_COLORS, kernel_size=1)  # -> pool over y (H)

    def forward(self, fmap: torch.Tensor) -> torch.Tensor:
        row = self.row(fmap).amax(dim=3)  # [B, 16, H]  (logit per y)
        col = self.col(fmap).amax(dim=2)  # [B, 16, W]  (logit per x)
        # joint[b, k, y, x] = row[b, k, y] + col[b, k, x]
        return row.unsqueeze(3) + col.unsqueeze(2)  # [B, 16, H, W]


class Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.body = ConvNeXtBody(hidden=128, blocks=4)
        self.ac = ActionColorPool(self.body.out_channels)
        self.click = SeparableClickHead(self.body.out_channels)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        f = self.body(x)
        action, color = self.ac(f)
        return action, color, self.click(f)


def build() -> Model:
    return Model()
