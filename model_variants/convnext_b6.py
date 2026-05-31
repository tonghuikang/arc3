"""convnext_b6 — frontier-exploit variant (auto-written file).

ConvNeXt hidden 96, 6 blocks.
"""

from __future__ import annotations

from model_variants.convnext import ConvNeXt


def build() -> ConvNeXt:
    return ConvNeXt(hidden=96, blocks=6)
