"""Autoresearch model variants — drop-in architectures, auto-discovered.

To add a model: drop a new module in this package that defines a top-level
``build() -> nn.Module`` whose ``forward(x)`` returns the three policy heads
(``action_logits``, ``color_logits``, ``click_logits``; see ``mlp.py``). That's
it — it becomes a selectable variant named by its filename, with **no other file
to edit**. Launch it with::

    uv run modal run kernels/autoresearch/model_variants/run.py --variant <filename>
    uv run modal run kernels/autoresearch/model_variants/run.py --variant all

``list_variants`` discovers every module here that exposes ``build()``;
``get_model(name)`` imports that module and calls it. The shared loss lives in
``model.py`` and is re-exported for convenience.
"""

from __future__ import annotations

import importlib
import pkgutil

from model import compute_loss, masked_action_logits

# Infra modules in this package that are not model variants.
_NOT_VARIANTS = {"run", "mlp"}


def list_variants() -> list[str]:
    """Every module here that defines ``build()`` — i.e. every model variant."""
    names: list[str] = []
    for info in pkgutil.iter_modules(__path__):
        name = info.name
        if name.startswith("_") or name in _NOT_VARIANTS:
            continue
        module = importlib.import_module(f"{__name__}.{name}")
        if callable(getattr(module, "build", None)):
            names.append(name)
    return sorted(names)


def get_model(name: str):
    """Instantiate the variant module ``name`` via its ``build()``."""
    module = importlib.import_module(f"{__name__}.{name}")
    build = getattr(module, "build", None)
    if not callable(build):
        raise ValueError(
            f"model_variants.{name} has no build(); not a variant. "
            f"Available: {list_variants()}"
        )
    return build()


__all__ = ["compute_loss", "get_model", "list_variants", "masked_action_logits"]
