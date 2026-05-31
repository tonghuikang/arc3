"""convnext_h160 — frontier-exploit variant (auto-written file).

ConvNeXt hidden 160, 4 blocks. Fills the width gap between h128 (current best)
and h192, pushing the winning width-scaling direction.
"""

from __future__ import annotations

from model_variants.convnext import ConvNeXt


def build() -> ConvNeXt:
    return ConvNeXt(hidden=160, blocks=4)
