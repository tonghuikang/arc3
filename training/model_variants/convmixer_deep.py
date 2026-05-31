"""convmixer_deep — ConvMixer, deeper (8 blocks)."""

from __future__ import annotations

from model_variants.convmixer import ConvMixer


def build() -> ConvMixer:
    return ConvMixer(dim=96, depth=8)
