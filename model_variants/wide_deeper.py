"""wide_deeper variant — wide and three layers deep (hidden 128, 3 layers)."""

from __future__ import annotations

from model_variants.mlp import PerCoordPolicy


def build() -> PerCoordPolicy:
    return PerCoordPolicy(hidden=128, hidden_layers=3)
