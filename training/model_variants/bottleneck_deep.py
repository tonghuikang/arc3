"""bottleneck_deep — frontier-exploit variant (auto-written file).

Bottleneck hidden 128, mid 32, 8 blocks.
"""

from __future__ import annotations

from model_variants.bottleneck import Bottleneck


def build() -> Bottleneck:
    return Bottleneck(hidden=128, mid=32, blocks=8)
