"""convnext_dw11 — ConvNeXt (h128 sweet spot) with an 11x11 depthwise kernel.

Very large spatial context. Hidden 128, 4 blocks, SiLU.
"""

from __future__ import annotations

from model_variants.convnext_flex import FlexConvNeXt


def build() -> FlexConvNeXt:
    return FlexConvNeXt(hidden=128, blocks=4, kernel=11)
