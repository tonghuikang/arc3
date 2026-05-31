"""convmixer_h160_d8 — frontier-exploit variant (auto-written file).

ConvMixer dim 160, 8 blocks.
"""

from __future__ import annotations

from model_variants.convmixer import ConvMixer


def build() -> ConvMixer:
    return ConvMixer(dim=160, depth=8)
