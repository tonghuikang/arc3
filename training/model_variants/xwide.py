"""xwide variant — a very wide single-layer per-coordinate MLP (hidden 256, 1 layer)."""

from __future__ import annotations

from model_variants.mlp import PerCoordPolicy


def build() -> PerCoordPolicy:
    return PerCoordPolicy(hidden=256, hidden_layers=1)
