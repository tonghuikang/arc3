"""se_resnet_h128_b8 — frontier-exploit variant (auto-written file).

SE-ResNet hidden 128, 8 blocks.
"""

from __future__ import annotations

from model_variants.se_resnet import SEResNet


def build() -> SEResNet:
    return SEResNet(hidden=128, blocks=8)
