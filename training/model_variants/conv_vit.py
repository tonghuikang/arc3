"""conv_vit variant — a conv-stem + transformer hybrid.

Two full-resolution 3x3 conv layers mix each cell with its neighbours *before*
the board is tokenised, then self-attention mixes globally. This local-then-
global recipe (cf. early-conv ViTs) often trains more stably and captures small
spatial motifs that pure patch embedding can blur across a 4x4 patch boundary.
"""

from __future__ import annotations

from model_variants.vit import SpatialTransformer


def build() -> SpatialTransformer:
    return SpatialTransformer(dim=128, depth=4, heads=4, patch=4, conv_stem=2)
