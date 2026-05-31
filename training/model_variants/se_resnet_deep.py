"""se_resnet_deep — SE-ResNet, deeper (8 blocks)."""

from __future__ import annotations

from model_variants.se_resnet import SEResNet


def build() -> SEResNet:
    return SEResNet(hidden=96, blocks=8)
