"""convnext_dw5 — ConvNeXt (h128 sweet spot) with a 5x5 depthwise kernel.

Probes how much spatial context the winning block wants: smaller (5x5) than the
default 7x7. Hidden 128, 4 blocks, SiLU.
"""

from __future__ import annotations

from model_variants.convnext_flex import FlexConvNeXt


def build() -> FlexConvNeXt:
    return FlexConvNeXt(hidden=128, blocks=4, kernel=5)
