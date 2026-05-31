"""convnext_h64_b4 — frontier-exploit variant (auto-written file).

ConvNeXt hidden 64, 4 blocks.
"""

from __future__ import annotations

from model_variants.convnext import ConvNeXt


def build() -> ConvNeXt:
    return ConvNeXt(hidden=64, blocks=4)
