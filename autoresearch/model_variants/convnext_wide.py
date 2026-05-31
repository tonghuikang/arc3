"""convnext_wide — the winning ConvNeXt design, wider (hidden 160)."""

from __future__ import annotations

from model_variants.convnext import ConvNeXt


def build() -> ConvNeXt:
    return ConvNeXt(hidden=160, blocks=4)
