"""wide_deep variant — wider and deeper per-coordinate MLP (hidden 128, 2 layers)."""

from __future__ import annotations

from model_variants.mlp import PerCoordPolicy


def build() -> PerCoordPolicy:
    return PerCoordPolicy(hidden=128, hidden_layers=2)
