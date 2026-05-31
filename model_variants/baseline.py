"""baseline variant — identical to the runtime ``model.Model`` (hidden 64, 1 layer)."""

from __future__ import annotations

from model_variants.mlp import PerCoordPolicy


def build() -> PerCoordPolicy:
    return PerCoordPolicy(hidden=64, hidden_layers=1)
