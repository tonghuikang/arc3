"""se_resnet_h192_b4 — frontier-exploit variant (auto-written file).

SE-ResNet hidden 192, 4 blocks.
"""

from __future__ import annotations

from model_variants.se_resnet import SEResNet


def build() -> SEResNet:
    return SEResNet(hidden=192, blocks=4)
