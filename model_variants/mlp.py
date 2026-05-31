"""Shared base architecture for autoresearch model variants.

A *variant* is any module in this package that defines a top-level ``build()``
returning an ``nn.Module`` whose ``forward(x)`` yields the three policy heads —
``action_logits [B, 6]``, ``color_logits [B, 16]``, ``click_logits [B, 16, 64,
64]`` — so it trains with ``model.compute_loss``. Drop a new file with a
``build()`` into this package and it is picked up automatically (no other file
changes); see ``__init__.get_model`` / ``list_variants``.

Most variants are just a :class:`PerCoordPolicy` with a different width/depth.
Feature-layout constants and the loss live in ``model.py``.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from model import HIDDEN, NUM_ACTIONS, NUM_COLORS, NUM_FEATURES

# 6 action planes + 16 colour planes + 16 per-colour click maps.
OUT_PLANES = NUM_ACTIONS + 2 * NUM_COLORS


class PerCoordPolicy(nn.Module):
    """Per-coordinate MLP (stacked 1x1 convs) emitting the three policy heads."""

    def __init__(
        self,
        hidden: int = 64,
        hidden_layers: int = 1,
        num_features: int = NUM_FEATURES,
    ) -> None:
        super().__init__()
        if hidden_layers < 1:
            raise ValueError("hidden_layers must be >= 1")
        self.hidden = hidden
        self.hidden_layers = hidden_layers
        self.num_features = num_features
        convs: list[nn.Module] = []
        in_c = num_features
        for _ in range(hidden_layers):
            convs.append(nn.Conv2d(in_c, hidden, kernel_size=1))
            in_c = hidden
        self.trunk = nn.ModuleList(convs)
        self.head = nn.Conv2d(in_c, OUT_PLANES, kernel_size=1)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = x
        for conv in self.trunk:
            h = F.silu(conv(h))
        out = self.head(h)  # [B, 6 + 16 + 16, H, W]
        action_planes = out[:, :NUM_ACTIONS]
        color_planes = out[:, NUM_ACTIONS : NUM_ACTIONS + NUM_COLORS]
        click_logits = out[:, NUM_ACTIONS + NUM_COLORS :]
        action_logits = action_planes.amax(dim=(2, 3))  # [B, 6]
        color_logits = color_planes.amax(dim=(2, 3))  # [B, 16]
        return action_logits, color_logits, click_logits

    @staticmethod
    def count_parameters(model: nn.Module) -> int:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    def runtime_state_dict(self) -> dict | None:
        """State dict re-keyed for the runtime ``model.Model``, or ``None``.

        Only the single-hidden-layer, default-width architecture maps onto the
        runtime model's ``conv1``/``conv2`` layout (``model.Model`` /
        ``my_agent``). Other variants have no runtime-compatible mapping.
        """
        if self.hidden_layers != 1 or self.hidden != HIDDEN:
            return None
        sd = self.state_dict()
        return {
            "conv1.weight": sd["trunk.0.weight"],
            "conv1.bias": sd["trunk.0.bias"],
            "conv2.weight": sd["head.weight"],
            "conv2.bias": sd["head.bias"],
        }
