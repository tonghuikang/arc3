"""wide variant — a wider single-layer per-coordinate MLP (hidden 128)."""

from __future__ import annotations

from model_variants.mlp import PerCoordPolicy


def build() -> PerCoordPolicy:
    return PerCoordPolicy(hidden=128, hidden_layers=1)
