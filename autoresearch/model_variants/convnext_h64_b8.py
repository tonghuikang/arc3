"""convnext_h64_b8 — frontier-exploit variant (auto-written file).

ConvNeXt hidden 64, 8 blocks.
"""

from __future__ import annotations

from model_variants.convnext import ConvNeXt


def build() -> ConvNeXt:
    return ConvNeXt(hidden=64, blocks=8)
