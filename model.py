"""Feature-layout constants and imitation-loss helpers for the autoresearch model.

The ``Model`` architecture itself lives in the training repo and is intentionally
not published here. This module exposes the 384-feature input-layout constants
plus the loss helpers (:func:`compute_loss`, :func:`masked_action_logits`) that
the ``model_variants/`` files reference.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

BOARD_SIZE = 64
NUM_COLORS = 16
NUM_ACTIONS = 6  # ACTION1..ACTION6; ACTION6 is click.
CLICK_ACTION_ID = 6
CLICK_ACTION_INDEX = CLICK_ACTION_ID - 1

# --- input feature layout: 384 binary features per coordinate ----------------
# Four 16-colour groups: latest frame, OR of the intermediate frames since the
# previous action, level-OR (all frames from level start to now), and the first
# frame of the level.
NUM_COLOR_GROUPS = 4
F_COLOR = NUM_COLOR_GROUPS * NUM_COLORS  # 64
# Per historical frame: 16 colour one-hots + 6 action one-hots (broadcast) +
# 1 "clicked cell" + 1 "changed cells" mask.
PER_FRAME = NUM_COLORS + NUM_ACTIONS + 2  # 24
NUM_PREV_FRAMES = 6  # the 6 frames before the current frame
F_PREV = NUM_PREV_FRAMES * PER_FRAME  # 144
NUM_LASTLEVEL_FRAMES = 2  # first and last frame of the previous level
F_LASTLEVEL = NUM_LASTLEVEL_FRAMES * PER_FRAME  # 48
POS_PLANES = BOARD_SIZE  # 2**6 == 64 planes per axis (one-hot over x / over y)
F_POS = 2 * POS_PLANES  # 128
NUM_FEATURES = F_COLOR + F_PREV + F_LASTLEVEL + F_POS  # 384
assert NUM_FEATURES == 384, f"NUM_FEATURES={NUM_FEATURES}, expected 384"

HIDDEN = 64  # keep it small for now


def masked_action_logits(
    action_logits: torch.Tensor, action_mask: torch.Tensor
) -> torch.Tensor:
    """Set illegal-action logits to a large negative value before CE/softmax."""
    if action_mask.dtype != torch.bool:
        action_mask = action_mask.to(torch.bool)
    return action_logits.masked_fill(~action_mask, -1.0e9)


def compute_loss(
    action_logits: torch.Tensor,
    color_logits: torch.Tensor,
    click_logits: torch.Tensor,
    batch: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, dict[str, float]]:
    """Imitation loss for the three independent heads.

    - **action**: cross-entropy over the 6 legal-masked actions, for every
      sample.
    - **colour**: cross-entropy over the 16 colours, *only for click samples*
      (no loss when the target action is not a click).
    - **coordinate**: cross-entropy over the 64x64 click map, *only for click
      samples* and *only on the map of that sample's target colour* — the 15
      other per-colour maps receive no gradient, and non-click samples
      contribute nothing.

    ``batch`` is a dict of stacked tensors with keys ``target_action`` ``[B]``,
    ``target_color`` ``[B]``, ``target_click_x`` ``[B]``, ``target_click_y``
    ``[B]``, ``is_click`` ``[B]`` bool, and ``action_mask`` ``[B, 6]`` bool.
    Returns the scalar total loss and a dict of detached per-head loss values.
    """
    action_loss = F.cross_entropy(
        masked_action_logits(action_logits, batch["action_mask"]),
        batch["target_action"],
        reduction="mean",
    )

    is_click = batch["is_click"].to(torch.bool)
    if is_click.any():
        idx = is_click.nonzero(as_tuple=True)[0]
        target_color = batch["target_color"][idx]
        # Colour head: which of the 16 colours the click landed on.
        color_loss = F.cross_entropy(color_logits[idx], target_color, reduction="mean")
        # Coordinate head: only each click sample's target-colour map contributes.
        color_maps = click_logits[idx, target_color]  # [k, H, W]
        width = color_maps.size(-1)
        coord_logits = color_maps.reshape(idx.numel(), -1)
        coord_target = (
            batch["target_click_y"][idx] * width + batch["target_click_x"][idx]
        )
        coord_loss = F.cross_entropy(coord_logits, coord_target, reduction="mean")
    else:
        color_loss = color_logits.sum() * 0.0
        coord_loss = click_logits.sum() * 0.0

    total = action_loss + color_loss + coord_loss
    metrics = {
        "total": float(total.detach()),
        "action": float(action_loss.detach()),
        "color": float(color_loss.detach()),
        "coord": float(coord_loss.detach()),
    }
    return total, metrics
