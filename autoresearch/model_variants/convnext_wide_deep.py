"""convnext_wide_deep — frontier-exploit variant (auto-written file).

ConvNeXt hidden 160, 8 blocks (wide+deep).
"""

from __future__ import annotations

from model_variants.convnext import ConvNeXt


def build() -> ConvNeXt:
    return ConvNeXt(hidden=160, blocks=8)
