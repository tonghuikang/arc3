"""convnext_h64 — frontier-exploit variant (auto-written file).

ConvNeXt hidden 64, 6 blocks.
"""

from __future__ import annotations

from model_variants.convnext import ConvNeXt


def build() -> ConvNeXt:
    return ConvNeXt(hidden=64, blocks=6)
