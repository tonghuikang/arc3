"""convnext_h160_b6 — frontier-exploit variant (auto-written file).

ConvNeXt hidden 160, 6 blocks.
"""

from __future__ import annotations

from model_variants.convnext import ConvNeXt


def build() -> ConvNeXt:
    return ConvNeXt(hidden=160, blocks=6)
