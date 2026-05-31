"""dilated_h128 — frontier-exploit variant (auto-written file).

SiLU dilated CNN, hidden 128, dilations 1..16.
"""

from __future__ import annotations

from model_variants.dilated_silu import DilatedCNNSiLU


def build() -> DilatedCNNSiLU:
    return DilatedCNNSiLU(hidden=128, dilations=(1, 2, 4, 8, 16))
