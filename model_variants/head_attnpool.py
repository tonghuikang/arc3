"""head_attnpool — convnext_h128 body + attention pooling for action/color.

The shared head reduces the action/color planes with amax (one winning cell). A
hard max throws away distributed evidence; here each head channel computes a
softmax spatial attention over its own value map and takes the attended mean.
Click stays a 1x1 conv — an isolated A/B on the pooling, not the click decoder.
Body: convnext_h128.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from model import NUM_ACTIONS, NUM_COLORS
from model_variants.cvx_body import ConvNeXtBody


class AttnPool(nn.Module):
    def __init__(self, in_ch: int) -> None:
        super().__init__()
        n = NUM_ACTIONS + NUM_COLORS
        self.val = nn.Conv2d(in_ch, n, kernel_size=1)
        self.att = nn.Conv2d(in_ch, n, kernel_size=1)
        self.click = nn.Conv2d(in_ch, NUM_COLORS, kernel_size=1)

    def forward(
        self, fmap: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        val = self.val(fmap)  # [B, 22, H, W]
        att = self.att(fmap)
        w = torch.softmax(att.flatten(2), dim=2).view_as(att)  # over space
        pooled = (val * w).sum(dim=(2, 3))  # [B, 22]
        action = pooled[:, :NUM_ACTIONS]
        color = pooled[:, NUM_ACTIONS:]
        return action, color, self.click(fmap)


class Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.body = ConvNeXtBody(hidden=128, blocks=4)
        self.head = AttnPool(self.body.out_channels)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.head(self.body(x))


def build() -> Model:
    return Model()
