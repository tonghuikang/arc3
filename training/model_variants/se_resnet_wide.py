"""se_resnet_wide — frontier-exploit variant (auto-written file).

SE-ResNet hidden 128, 6 blocks.
"""

from __future__ import annotations

from model_variants.se_resnet import SEResNet


def build() -> SEResNet:
    return SEResNet(hidden=128, blocks=6)
