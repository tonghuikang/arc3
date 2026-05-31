#!/usr/bin/env python
"""Build metrics_export/index.json for the training-metrics dashboard.

metrics.html loads this compact index for the Frontier and Table views (which
need every run at once); the Model view then fetches each run's full
metrics_export/<run_id>.json on demand. So the index only carries the few
fields those all-runs views read — not the per-step training curves.

Run it any time runs are added/changed:

    python training/build_metrics_index.py

It scans training/metrics_export/*.json and rewrites metrics_export/index.json.
"""

from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXPORT = HERE / "metrics_export"


def summarize(rec: dict) -> dict:
    """The minimal per-run record the Frontier + Table views consume."""
    rows = rec.get("train") or rec.get("steps") or []
    val = rec.get("val") or []
    last_val = val[-1] if val else None
    return {
        "run_id": rec.get("run_id"),
        "status": rec.get("status"),
        "num_params": rec.get("num_params"),
        "steps_total": rec.get("steps_total"),
        "last_step": (rows[-1].get("step") if rows else 0),
        "config": {"variant": (rec.get("config") or {}).get("variant")},
        # Frontier (scatter + frontier staircase) and Table only need the
        # final validation point: x = wall-clock time, y = total loss.
        "val": (
            [{"time": last_val.get("time"), "total": last_val.get("total")}]
            if last_val
            else []
        ),
    }


def main() -> None:
    summaries: dict[str, dict] = {}
    for f in sorted(EXPORT.glob("*.json")):
        if f.name == "index.json":
            continue
        rec = json.loads(f.read_text())
        rid = rec.get("run_id") or f.stem
        summaries[rid] = summarize(rec)
    runs = sorted(summaries)
    out = EXPORT / "index.json"
    out.write_text(json.dumps({"runs": runs, "all": summaries}, separators=(",", ":")))
    print(f"wrote {out.relative_to(HERE)} ({len(runs)} runs, {out.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
