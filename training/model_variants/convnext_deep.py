"""convnext_deep — the winning ConvNeXt design, deeper (8 blocks).

Exploits the current frontier leader (convnext) by adding depth at the same
width. Reuses the base ConvNeXt class unchanged (new file, new build()).
"""

from __future__ import annotations

from model_variants.convnext import ConvNeXt


def build() -> ConvNeXt:
    return ConvNeXt(hidden=96, blocks=8)
