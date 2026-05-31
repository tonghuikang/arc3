"""repvgg_wide — frontier-exploit variant (auto-written file).

RepVGG hidden 160, 5 layers.
"""

from __future__ import annotations

from model_variants.repvgg import RepVGG


def build() -> RepVGG:
    return RepVGG(hidden=160, layers=5)
