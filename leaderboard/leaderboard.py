# Leaderboard listener for the ARC Prize 2026 / ARC-AGI-3 Kaggle competition.
#
# Polls the public Kaggle leaderboard every minute, records each team's score
# history into persistent Modal Dicts, and serves it as JSON for leaderboard.html.
#
#   uv run modal deploy leaderboard/leaderboard.py
#   uv run modal app logs arc3-leaderboard-monitor
#
# Requires a Modal secret named "kaggle" exposing KAGGLE_API_TOKEN (the verbatim
# contents of your ~/.kaggle/kaggle.json), e.g.:
#   uv run modal secret create kaggle KAGGLE_API_TOKEN="$(cat ~/.kaggle/kaggle.json)"

import os
from datetime import datetime

import modal

COMPETITION = "arc-prize-2026-arc-agi-3"

app = modal.App("arc3-leaderboard-monitor")

image = modal.Image.debian_slim(python_version="3.11").pip_install("kaggle")

# Persistent Dict storage
positions_dict = modal.Dict.from_name("arc3-leaderboard-positions", create_if_missing=True)
history_dict = modal.Dict.from_name("arc3-leaderboard-history", create_if_missing=True)

# The full per-team history is stored as a single blob under this one key. Modal Dict
# access is a network round-trip per key, so the previous one-key-per-team layout cost
# ~989 reads in the loop plus a ~989-key dict(history_dict) materialization every minute.
# A single blob collapses that to one read + one write per run.
HISTORY_KEY = "__all_history__"


def parse_leaderboard_csv(csv_content: str) -> list[dict]:
    """Parse kaggle leaderboard CSV output into structured data."""
    import csv
    import io

    entries = []
    reader = csv.DictReader(io.StringIO(csv_content))

    for row in reader:
        entries.append(
            {
                "team_id": row.get("TeamId", row.get("teamId", "")),
                "team_name": row.get("TeamName", row.get("teamName", "")),
                "submission_date": row.get(
                    "LastSubmissionDate",
                    row.get("SubmissionDate", row.get("submissionDate", "")),
                ),
                "score": row.get("Score", row.get("score", "")),
                "submission_count": row.get(
                    "SubmissionCount", row.get("submissionCount", "")
                ),
            }
        )
    return entries


@app.function(
    image=image,
    secrets=[modal.Secret.from_name("kaggle")],
    schedule=modal.Cron("* * * * *"),  # Every minute
    timeout=20 * 60,
    max_containers=1,
)
def check_leaderboard():
    import subprocess
    from pathlib import Path

    # Setup kaggle credentials from KAGGLE_API_TOKEN
    kaggle_token = os.environ.get("KAGGLE_API_TOKEN")
    if kaggle_token:
        kaggle_dir = Path.home() / ".kaggle"
        kaggle_dir.mkdir(exist_ok=True)
        kaggle_json = kaggle_dir / "kaggle.json"
        kaggle_json.write_text(kaggle_token)
        kaggle_json.chmod(0o600)

    import time

    _t0 = time.perf_counter()
    now = datetime.utcnow()
    print(f"\n{'=' * 60}")
    print(f"Leaderboard check at {now.isoformat()} UTC")
    print(f"{'=' * 60}\n")

    # Download full leaderboard to a temp file (clean up first to avoid stale files)
    download_dir = Path("/tmp/leaderboard")
    import shutil

    if download_dir.exists():
        shutil.rmtree(download_dir)
    download_dir.mkdir(exist_ok=True)

    result = subprocess.run(
        [
            "kaggle",
            "competitions",
            "leaderboard",
            COMPETITION,
            "--download",
            "--path",
            str(download_dir),
        ],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        print(f"Error: {result.stderr}")
        return result.stdout

    # Extract ZIP if present, then read CSV
    import zipfile

    for zip_file in download_dir.glob("*.zip"):
        with zipfile.ZipFile(zip_file, "r") as zf:
            zf.extractall(download_dir)

    csv_files = list(download_dir.glob("*.csv"))
    if not csv_files:
        print(f"No CSV file found in {download_dir}")
        print(f"Files: {list(download_dir.iterdir())}")
        return ""
    csv_file = csv_files[0]

    csv_content = csv_file.read_text()
    entries = parse_leaderboard_csv(csv_content)

    _t_parsed = time.perf_counter()

    # Load the entire history in ONE read instead of one round-trip per team.
    # First run after the layout switch: migrate the legacy per-team keys (one slow
    # dict() materialization, once); afterwards it's a single blob get.
    all_history = history_dict.get(HISTORY_KEY)
    migrated = all_history is None
    if migrated:
        all_history = {k: v for k, v in dict(history_dict).items() if k != HISTORY_KEY}
    _t_loaded = time.perf_counter()

    # Build leaderboard_positions: [team_id, team_name, position, submission_count, score]
    leaderboard_positions = []
    new_submissions = 0
    updated_scores = 0

    for position, entry in enumerate(entries, start=1):
        team_id = entry["team_id"]
        if not team_id:
            continue

        team_name = entry["team_name"]
        score = entry["score"]
        submission_count = entry["submission_count"]
        submission_date_str = entry["submission_date"]

        # Add to positions list
        leaderboard_positions.append(
            [team_id, team_name, position, submission_count, score]
        )

        # Parse submission time to create date key and calculate minutes_taken
        if submission_date_str:
            # Format: "2026-05-31 01:23:45.123456" -> "2026-05-31"
            date_str_clean = submission_date_str.split(".")[0]
            sub_time = datetime.strptime(date_str_clean, "%Y-%m-%d %H:%M:%S")
            date_key = sub_time.strftime("%Y-%m-%d")
            submission_time_formatted = sub_time.strftime("%Y-%m-%d-%H-%M")

            # Calculate minutes_taken as time since submission (relative to cron run time)
            delta = now - sub_time
            minutes_taken = int(delta.total_seconds() / 60)

            # Get existing history for this team (in-memory; no network round-trip)
            existing_entries = all_history.get(team_id, {})

            # Find an existing entry for this submission time, if any.
            matched_key = None
            for existing_date_key, existing_data in existing_entries.items():
                if existing_data.get("submission_time") == submission_time_formatted:
                    matched_key = existing_date_key
                    break

            if matched_key is None:
                # Genuinely new submission time -> append a new entry.
                # If there's already an entry for this date, use a unique key.
                final_date_key = date_key
                if date_key in existing_entries:
                    # Append a counter to make it unique
                    counter = 1
                    new_key = f"{date_key}-{counter}"
                    while new_key in existing_entries:
                        counter += 1
                        new_key = f"{date_key}-{counter}"
                    final_date_key = new_key

                existing_entries[final_date_key] = {
                    "score": score,
                    "minutes_taken": minutes_taken,
                    "submission_time": submission_time_formatted,
                }
                all_history[team_id] = existing_entries
                new_submissions += 1
            elif existing_entries[matched_key].get("score") != score:
                # Same submission, but Kaggle's (best) score changed since we first
                # recorded it -- scores lag/recompute on Kaggle, so update in place
                # rather than dropping the improvement (the old code never did this).
                existing_entries[matched_key]["score"] = score
                all_history[team_id] = existing_entries
                updated_scores += 1

    _t_looped = time.perf_counter()

    # Persist history as one blob (not per-team), but only when it actually changed
    # (or on the first-run migration). get_history serves cached_response, which is
    # rewritten every run since positions shift, and embeds the current history.
    if migrated or new_submissions or updated_scores:
        history_dict[HISTORY_KEY] = all_history
    positions_dict["cached_response"] = {
        "positions": leaderboard_positions,
        "history": all_history,
    }

    _t_written = time.perf_counter()
    print(
        f"TIMING: download+parse={_t_parsed - _t0:.2f}s | "
        f"load_history={_t_loaded - _t_parsed:.2f}s | "
        f"loop(in-memory)={_t_looped - _t_loaded:.2f}s | "
        f"write={_t_written - _t_looped:.2f}s | "
        f"total={_t_written - _t0:.2f}s"
    )

    print(f"Stored {len(leaderboard_positions)} teams in positions")
    print(f"New submissions recorded: {new_submissions}")
    print(f"Scores updated (late rescoring): {updated_scores}")

    # Print top 10 for quick reference
    print("\nTop 10:")
    print(f"{'Pos':>4} {'Team':<25} {'Score':>6} {'Subs':>5}")
    print(f"{'-' * 4} {'-' * 25} {'-' * 6} {'-' * 5}")
    for pos in leaderboard_positions[:10]:
        team_id, team_name, position, sub_count, score = pos
        print(f"{position:>4} {team_name[:25]:<25} {score:>6} {sub_count:>5}")

    return result.stdout


@app.function(max_containers=1)
@modal.fastapi_endpoint(method="GET")
def get_history():
    """Return leaderboard positions and historical scores as JSON.

    Format:
    {
        "positions": [[team_id, team_name, position, submission_count, score], ...],
        "history": {team_id: {date: [score, minutes_taken]}}
    }
    """
    from fastapi.responses import JSONResponse

    # Single lookup - cached by cron job
    cached = positions_dict.get("cached_response", {"positions": [], "history": {}})

    # Transform history to compact array format: [score, minutes_taken]
    all_history = cached.get("history", {})
    compact_history = {}
    for team_id, entries in all_history.items():
        compact_entries = {}
        for date_key, data in entries.items():
            compact_entries[date_key] = [data["score"], data["minutes_taken"]]
        if compact_entries:
            compact_history[team_id] = compact_entries

    return JSONResponse(
        content={"positions": cached.get("positions", []), "history": compact_history},
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, OPTIONS",
            "Access-Control-Allow-Headers": "*",
        },
    )
