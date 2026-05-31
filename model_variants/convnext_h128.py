"""convnext_h128 — frontier-exploit variant (auto-written file).

ConvNeXt hidden 128, 4 blocks.
"""

from __future__ import annotations

from model_variants.convnext import ConvNeXt


def build() -> ConvNeXt:
    return ConvNeXt(hidden=128, blocks=4)
