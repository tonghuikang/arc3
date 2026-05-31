"""deep variant — two hidden layers at the default width (hidden 64, 2 layers)."""

from __future__ import annotations

from model_variants.mlp import PerCoordPolicy


def build() -> PerCoordPolicy:
    return PerCoordPolicy(hidden=64, hidden_layers=2)
