"""vit_small variant — a cheaper spatial transformer (coarser patches, shallower).

Patch 8 -> 8x8=64 tokens (vs 256 for the default ViT), so attention is ~16x
lighter; a good fast baseline for the transformer family.
"""

from __future__ import annotations

from model_variants.vit import SpatialTransformer


def build() -> SpatialTransformer:
    return SpatialTransformer(dim=96, depth=3, heads=4, patch=8)
