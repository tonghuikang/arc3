"""convnext_h192_b8 — frontier-exploit variant (auto-written file).

ConvNeXt hidden 192, 8 blocks.
"""

from __future__ import annotations

from model_variants.convnext import ConvNeXt


def build() -> ConvNeXt:
    return ConvNeXt(hidden=192, blocks=8)
