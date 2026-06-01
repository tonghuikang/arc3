#!/bin/bash

# Serve the repo root over HTTP so the static dashboards read their local
# metrics files via fetch() (file:// blocks those requests).
#   http://localhost:$PORT/autoresearch/index.html   autoresearch graph
#   http://localhost:$PORT/leaderboard/leaderboard.html            leaderboard (ARC-AGI-3)
#   http://localhost:$PORT/leaderboard/leaderboard.html?comp=arc2  leaderboard (ARC-AGI-2)
PORT=4839

# Kill any existing process on the port
PID=$(lsof -t -i :$PORT 2>/dev/null)
if [ -n "$PID" ]; then
    echo "Killing existing process on port $PORT (PID: $PID)"
    kill $PID 2>/dev/null
    sleep 1
fi

echo "Starting server on http://localhost:$PORT/"
echo "  autoresearch: http://localhost:$PORT/autoresearch/index.html?view=model"
echo "  leaderboard (ARC-AGI-3): http://localhost:$PORT/leaderboard/leaderboard.html"
echo "  leaderboard (ARC-AGI-2): http://localhost:$PORT/leaderboard/leaderboard.html?comp=arc2"
nohup uv run python -m http.server $PORT > /dev/null 2>&1 &
echo "Server running in background (PID: $!)"
