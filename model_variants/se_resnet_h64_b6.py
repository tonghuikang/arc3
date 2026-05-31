"""se_resnet_h64_b6 — frontier-exploit variant (auto-written file).

SE-ResNet hidden 64, 6 blocks.
"""

from __future__ import annotations

from model_variants.se_resnet import SEResNet


def build() -> SEResNet:
    return SEResNet(hidden=64, blocks=6)
