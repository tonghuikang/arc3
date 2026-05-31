"""convnext_h192 — frontier-exploit variant (auto-written file).

ConvNeXt hidden 192, 4 blocks.
"""

from __future__ import annotations

from model_variants.convnext import ConvNeXt


def build() -> ConvNeXt:
    return ConvNeXt(hidden=192, blocks=4)
