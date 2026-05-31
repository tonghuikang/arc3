"""convnext_h128_deep — frontier-exploit variant (auto-written file).

ConvNeXt hidden 128, 8 blocks.
"""

from __future__ import annotations

from model_variants.convnext import ConvNeXt


def build() -> ConvNeXt:
    return ConvNeXt(hidden=128, blocks=8)
