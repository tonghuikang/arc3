"""convnext_h144 — ConvNeXt fine-width probe just above the h128 sweet spot.

Hidden 144, 4 blocks, SiLU.
"""

from __future__ import annotations

from model_variants.convnext_flex import FlexConvNeXt


def build() -> FlexConvNeXt:
    return FlexConvNeXt(hidden=144, blocks=4)
