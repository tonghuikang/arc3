"""Shared output head for spatial backbones (not a variant — no ``build()``).

Any backbone that produces a dense ``[B, C, 64, 64]`` feature map can cap it with
:class:`SpatialHead` to emit the three policy heads the loss expects:

- ``action_logits [B, 6]`` and ``color_logits [B, 16]`` — max-pooled over space
  (each cell votes; the strongest vote wins), exactly like the conv ``baseline``;
- ``click_logits [B, 16, 64, 64]`` — the per-colour click map at full resolution.

Discovery imports this module but skips it (it defines no ``build()``), so it is
plumbing shared by ``dilated``/``resnet``/``unet``/``mixer``/``axial`` rather than
a model variant itself.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from model import NUM_ACTIONS, NUM_COLORS

OUT_PLANES = NUM_ACTIONS + 2 * NUM_COLORS  # 6 actions + 16 colours + 16 click maps


class SpatialHead(nn.Module):
    """1x1-conv a ``[B, C, 64, 64]`` map into (action, colour, click) heads."""

    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.proj = nn.Conv2d(in_channels, OUT_PLANES, kernel_size=1)

    def forward(
        self, fmap: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        out = self.proj(fmap)  # [B, 38, 64, 64]
        action_logits = out[:, :NUM_ACTIONS].amax(dim=(2, 3))  # [B, 6]
        color_logits = out[:, NUM_ACTIONS : NUM_ACTIONS + NUM_COLORS].amax(dim=(2, 3))
        click_logits = out[:, NUM_ACTIONS + NUM_COLORS :]  # [B, 16, 64, 64]
        return action_logits, color_logits, click_logits
