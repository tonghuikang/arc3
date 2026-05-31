"""convnext_e8 — ConvNeXt (h128 sweet spot) with an 8x pointwise expansion.

Doubles the default 4x inverted-bottleneck width. Hidden 128, 4 blocks, SiLU.
"""

from __future__ import annotations

from model_variants.convnext_flex import FlexConvNeXt


def build() -> FlexConvNeXt:
    return FlexConvNeXt(hidden=128, blocks=4, expansion=8)
