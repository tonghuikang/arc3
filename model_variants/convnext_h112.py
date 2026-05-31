"""convnext_h112 — ConvNeXt fine-width probe just below the h128 sweet spot.

Hidden 112, 4 blocks, SiLU.
"""

from __future__ import annotations

from model_variants.convnext_flex import FlexConvNeXt


def build() -> FlexConvNeXt:
    return FlexConvNeXt(hidden=112, blocks=4)
