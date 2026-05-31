"""convmixer_h128_d5 — frontier-exploit variant (auto-written file).

ConvMixer dim 128, 5 blocks.
"""

from __future__ import annotations

from model_variants.convmixer import ConvMixer


def build() -> ConvMixer:
    return ConvMixer(dim=128, depth=5)
