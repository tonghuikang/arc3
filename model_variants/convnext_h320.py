"""convnext_h320 — frontier-exploit variant (auto-written file).

ConvNeXt hidden 320, 4 blocks. Pushes width past the h256 point to probe where
the width-scaling curve flattens or reverses.
"""

from __future__ import annotations

from model_variants.convnext import ConvNeXt


def build() -> ConvNeXt:
    return ConvNeXt(hidden=320, blocks=4)
