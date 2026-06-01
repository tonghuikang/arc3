# ARC Prize 2026 Leaderboard Listener

A leaderboard monitor for the two [ARC Prize 2026](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3)
Kaggle tracks, modeled on [aimo3's `leaderboard.*`](https://github.com/tonghuikang/aimo3):

| Track | Competition | View |
|-------|-------------|------|
| **ARC-AGI-3** | [`arc-prize-2026-arc-agi-3`](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3) | `leaderboard.html` (default) |
| **ARC-AGI-2** | [`arc-prize-2026-arc-agi-2`](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2) | `leaderboard.html?comp=arc2` |

- **`leaderboard.py`** — a single [Modal](https://modal.com) app (`arc3-leaderboard-monitor`). One
  per-minute cron polls **both** public Kaggle leaderboards via the `kaggle` CLI in one run, records
  each team's score history into per-competition persistent Modal `Dict`s (`arc3-*` and `arc2-*`), and
  exposes one `get_history` JSON endpoint that returns a track by `?comp=arc2` / `?comp=arc3`. A failure
  on one track is isolated so it never blocks the other.
- **`leaderboard.html`** — a single viewer (teams × days, with running-high-score highlighting and
  public/private sort) that fetches the live endpoint and auto-refreshes each minute. Pick a track
  with `?comp=arc3` (default) or `?comp=arc2`, or use the switcher in the header.

The app is named `arc3-leaderboard-monitor` (rather than something track-neutral) so the original
ARC-AGI-3 endpoint URL keeps working unchanged now that it also serves ARC-AGI-2. The `arc3-`/`arc2-`
`Dict`s persist by name independently of the app, so the accumulated history survives.

## Deploy

Requires a Modal secret named `kaggle` exposing `KAGGLE_API_TOKEN` (the verbatim contents of your
`~/.kaggle/kaggle.json`). Create it once if you don't have it:

```sh
uv run modal secret create kaggle KAGGLE_API_TOKEN="$(cat ~/.kaggle/kaggle.json)"
```

Then deploy (a single app, a single cron, both tracks):

```sh
uv run modal deploy leaderboard/leaderboard.py
uv run modal app logs arc3-leaderboard-monitor          # follow the cron
```

The deploy prints the `get_history` URL. The currently deployed endpoint is:

```
https://tonghuikang--arc3-leaderboard-monitor-get-history.modal.run            # ARC-AGI-3
https://tonghuikang--arc3-leaderboard-monitor-get-history.modal.run?comp=arc2  # ARC-AGI-2
```

It is baked into `leaderboard.html` as `ENDPOINT`. If you redeploy under a different Modal workspace,
update that constant (or set a track's `endpoint` to `''` to read a static `leaderboard.jsonl` /
`leaderboard2.jsonl` snapshot from this folder instead).

## Data format

`get_history` returns, for the requested track:

```json
{
  "positions": [[team_id, team_name, position, submission_count, score], ...],
  "history":   {"<team_id>": {"YYYY-MM-DD": [score, minutes_taken], ...}, ...}
}
```

`leaderboard.html` also accepts a static `<track>.jsonl` (one team record per line, with optional
`final_rank`/`final_score`) for an archived post-competition snapshot.
