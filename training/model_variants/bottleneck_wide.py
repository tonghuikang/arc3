"""bottleneck_wide — frontier-exploit variant (auto-written file).

Bottleneck hidden 160, mid 40, 6 blocks.
"""

from __future__ import annotations

from model_variants.bottleneck import Bottleneck


def build() -> Bottleneck:
    return Bottleneck(hidden=160, mid=40, blocks=6)
