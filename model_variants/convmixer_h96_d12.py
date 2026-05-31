"""convmixer_h96_d12 — frontier-exploit variant (auto-written file).

ConvMixer dim 96, 12 blocks.
"""

from __future__ import annotations

from model_variants.convmixer import ConvMixer


def build() -> ConvMixer:
    return ConvMixer(dim=96, depth=12)
