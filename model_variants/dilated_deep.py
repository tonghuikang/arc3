"""dilated_deep — SiLU dilated CNN with an extra octave (dilations up to 32)."""

from __future__ import annotations

from model_variants.dilated_silu import DilatedCNNSiLU


def build() -> DilatedCNNSiLU:
    return DilatedCNNSiLU(hidden=96, dilations=(1, 2, 4, 8, 16, 32))
