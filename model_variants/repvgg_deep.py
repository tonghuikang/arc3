"""repvgg_deep — frontier-exploit variant (auto-written file).

RepVGG hidden 96, 8 layers.
"""

from __future__ import annotations

from model_variants.repvgg import RepVGG


def build() -> RepVGG:
    return RepVGG(hidden=96, layers=8)
