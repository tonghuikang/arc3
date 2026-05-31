"""vit_deep variant — a deeper, wider spatial transformer (8 layers, 8 heads).

Same fine 4x4 patches as the default ViT (256 tokens) but twice as deep, to test
whether more attention depth buys better cross-cell reasoning.
"""

from __future__ import annotations

from model_variants.vit import SpatialTransformer


def build() -> SpatialTransformer:
    return SpatialTransformer(dim=128, depth=8, heads=8, patch=4)
