# ARC-AGI-3 Leaderboard Listener

A leaderboard monitor for the [ARC Prize 2026 / ARC-AGI-3](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/)
Kaggle competition, modeled on [aimo3's `leaderboard.*`](https://github.com/tonghuikang/aimo3).

- **`leaderboard.py`** — a [Modal](https://modal.com) app (`arc3-leaderboard-monitor`). A cron
  polls the public Kaggle leaderboard every minute via the `kaggle` CLI, records each team's score
  history into persistent Modal `Dict`s, and exposes a `get_history` JSON endpoint.
- **`leaderboard.html`** — a viewer (teams × days, with running-high-score highlighting and
  public/private sort) that fetches the `get_history` endpoint live and auto-refreshes each minute.

## Deploy

Requires a Modal secret named `kaggle` exposing `KAGGLE_API_TOKEN` (the verbatim contents of your
`~/.kaggle/kaggle.json`). Create it once if you don't have it:

```sh
uv run modal secret create kaggle KAGGLE_API_TOKEN="$(cat ~/.kaggle/kaggle.json)"
```

Then deploy:

```sh
uv run modal deploy leaderboard/leaderboard.py
uv run modal app logs arc3-leaderboard-monitor          # follow the cron
```

The deploy prints the `get_history` URL. The currently deployed endpoint is:

```
https://tonghuikang--arc3-leaderboard-monitor-get-history.modal.run
```

It is already baked into `leaderboard.html` as `ENDPOINT_URL`. If you redeploy under a different
Modal workspace, update that constant (or set it to `''` to read a static `leaderboard.jsonl`
snapshot from this folder instead).

## Data format

`get_history` returns:

```json
{
  "positions": [[team_id, team_name, position, submission_count, score], ...],
  "history":   {"<team_id>": {"YYYY-MM-DD": [score, minutes_taken], ...}, ...}
}
```

`leaderboard.html` also accepts a static `leaderboard.jsonl` (one team record per line, with optional
`final_rank`/`final_score`) for an archived post-competition snapshot.
