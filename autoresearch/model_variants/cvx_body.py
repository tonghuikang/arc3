"""cvx_body — the frontier-winning ConvNeXt backbone, head-less (NOT a variant).

The architecture search plateaued because every backbone hits the same per-term
loss floor; the unexplored lever is the *head* (the click coordinate term is ~half
the loss yet flows through a single 1x1 conv). These head experiments reuse the
proven convnext_h128 body and vary only the head. This module exposes the body as
a reusable map producer and defines no ``build()``, so it is never discovered as a
variant itself (same convention as ``heads.py``).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from model import NUM_ACTIONS, NUM_COLORS, NUM_FEATURES
from model_variants.convnext_flex import FlexConvNeXtBlock


class ActionColorPool(nn.Module):
    """Proven default action/color head: 1x1 conv -> amax-pool over space.

    Click heads that only want to vary the *click* decoder reuse this so the
    action/color path is held fixed (a clean A/B on the click term).
    """

    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.proj = nn.Conv2d(in_channels, NUM_ACTIONS + NUM_COLORS, kernel_size=1)

    def forward(self, fmap: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        o = self.proj(fmap)
        action = o[:, :NUM_ACTIONS].amax(dim=(2, 3))
        color = o[:, NUM_ACTIONS:].amax(dim=(2, 3))
        return action, color


class ConvNeXtBody(nn.Module):
    """convnext_h128 stem + blocks; forward -> dense ``[B, hidden, 64, 64]`` map."""

    def __init__(
        self,
        hidden: int = 128,
        blocks: int = 4,
        kernel: int = 7,
        expansion: int = 4,
    ) -> None:
        super().__init__()
        self.stem = nn.Conv2d(NUM_FEATURES, hidden, kernel_size=1)
        self.blocks = nn.Sequential(
            *[FlexConvNeXtBlock(hidden, kernel, expansion) for _ in range(blocks)]
        )
        self.out_channels = hidden

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(self.stem(x))
