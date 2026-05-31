"""convnext_dw9 — ConvNeXt (h128 sweet spot) with a 9x9 depthwise kernel.

Larger spatial context than the default 7x7. Hidden 128, 4 blocks, SiLU.
"""

from __future__ import annotations

from model_variants.convnext_flex import FlexConvNeXt


def build() -> FlexConvNeXt:
    return FlexConvNeXt(hidden=128, blocks=4, kernel=9)
