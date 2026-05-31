"""convnext_e6 — ConvNeXt (h128 sweet spot) with a 6x pointwise expansion.

Wider inverted-bottleneck MLP (default is 4x) for more channel-mixing capacity.
Hidden 128, 4 blocks, SiLU.
"""

from __future__ import annotations

from model_variants.convnext_flex import FlexConvNeXt


def build() -> FlexConvNeXt:
    return FlexConvNeXt(hidden=128, blocks=4, expansion=6)
