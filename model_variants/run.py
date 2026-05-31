#!/usr/bin/env python
"""Autoresearch launch surface — sweep model variants through the training harness.

This is the entrypoint you invoke. Every module in ``model_variants/`` that
defines ``build()`` is an architecture; drop a new file in and it is discovered
automatically (no other file to edit). Launch any subset — or the whole fleet:

    uv run modal run kernels/autoresearch/model_variants/run.py --variant baseline
    uv run modal run kernels/autoresearch/model_variants/run.py \
        --variant baseline,wide,deep            # a hand-picked sweep
    uv run modal run kernels/autoresearch/model_variants/run.py --variant all

Every selected variant is launched **in parallel** — each becomes its own Modal
GPU container (``train.spawn``), so the whole sweep runs concurrently rather than
one-at-a-time. Training always uses all (variation) games — autoresearch sweeps
*models*, not game subsets. It imports the reusable training harness from
``training/modal_train.py``; each job is a single epoch with exactly one
validation on the held-out non-variant games at the end. As each finishes we
download its checkpoint + ``metrics_<run>.json``. The harness never imports this
file, so there is no import cycle.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent  # .../model_variants
_AUTORESEARCH_DIR = _HERE.parent  # .../autoresearch
# Make the harness (training/modal_train.py) and model/model_variants importable.
sys.path.insert(0, str(_AUTORESEARCH_DIR / "training"))
sys.path.insert(0, str(_AUTORESEARCH_DIR))

import modal  # noqa: E402

from model_variants import list_variants  # noqa: E402
from modal_train import app, ckpt_volume, train  # noqa: E402


@app.local_entrypoint()
def main(
    only_won: bool = True,
    max_input_level: int = -1,
    batch_size: int = 64,
    lr: float = 1e-3,
    variant: str = "all",
    log_every: int = 1,
    num_workers: int = 4,
    run_id: str = "",
    out_dir: str = "kernels/autoresearch/training/runs",
    weight_decay: float = 0.01,
    schedule: str = "constant",
    warmup_frac: float = 0.0,
    tag: str = "",
) -> None:
    mil = None if max_input_level < 0 else max_input_level
    # --variant: "all" sweeps every discovered variant; or a comma list of names
    # (each is a module file in model_variants/ that defines build()).
    if variant.strip() == "all":
        variants = list_variants()
    else:
        variants = [v.strip() for v in variant.split(",") if v.strip()]
    if not variants:
        variants = list_variants()
    base_ts = int(time.time())

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Launch every variant CONCURRENTLY: spawn() returns a handle immediately
    # (non-blocking), so all Modal GPU jobs run in parallel — each in its own
    # container — instead of one-at-a-time. We collect results afterwards.
    launched: list[tuple[str, str, object]] = []
    for v in variants:
        # ``tag`` distinguishes recipe-sweep runs of the SAME variant (e.g.
        # convnext_h128 at several learning rates) so their run_ids don't collide.
        label = f"{v}-{tag}" if tag else v
        rid = run_id if (run_id and len(variants) == 1) else f"{label}-{base_ts}"
        call = train.spawn(
            run_id=rid,
            only_won=only_won,
            max_input_level=mil,
            batch_size=batch_size,
            lr=lr,
            variant=v,
            log_every=log_every,
            num_workers=num_workers,
            weight_decay=weight_decay,
            schedule=schedule,
            warmup_frac=warmup_frac,
        )
        launched.append((v, rid, call))
        print(f"[{v}] spawned run_id={rid} (call {call.object_id})")

    print(f"launched {len(launched)} run(s) in parallel; collecting results…")
    for v, rid, call in launched:
        try:
            result = call.get()
        except (modal.exception.Error, RuntimeError, OSError, ValueError) as exc:
            # one variant failing must not sink the rest of the parallel sweep
            print(f"[{v}] FAILED (run_id={rid}): {exc}")
            continue
        print(f"[{v}] train result:", json.dumps(result, indent=1, default=float))
        # read_file fetches the latest committed state directly; reload() is
        # container-only and must not be called from a local entrypoint.
        for name in (result["ckpt"], f"metrics_{rid}.json"):
            (out / name).write_bytes(b"".join(ckpt_volume.read_file(name)))
        print(f"downloaded {result['ckpt']} and metrics_{rid}.json to {out}/")
