"""convnext_h256_b4 — frontier-exploit variant (auto-written file).

ConvNeXt hidden 256, 4 blocks.
"""

from __future__ import annotations

from model_variants.convnext import ConvNeXt


def build() -> ConvNeXt:
    return ConvNeXt(hidden=256, blocks=4)
