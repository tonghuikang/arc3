"""convnext_b12 — frontier-exploit variant (auto-written file).

ConvNeXt hidden 96, 12 blocks.
"""

from __future__ import annotations

from model_variants.convnext import ConvNeXt


def build() -> ConvNeXt:
    return ConvNeXt(hidden=96, blocks=12)
