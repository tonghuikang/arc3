"""tiny variant — a narrow single-layer per-coordinate MLP (hidden 32, 1 layer)."""

from __future__ import annotations

from model_variants.mlp import PerCoordPolicy


def build() -> PerCoordPolicy:
    return PerCoordPolicy(hidden=32, hidden_layers=1)
