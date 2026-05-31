"""se_resnet_h160 — frontier-exploit variant (auto-written file).

SE-ResNet hidden 160, 4 blocks.
"""

from __future__ import annotations

from model_variants.se_resnet import SEResNet


def build() -> SEResNet:
    return SEResNet(hidden=160, blocks=4)
