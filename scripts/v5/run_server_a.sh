#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
SESSION="posttech-v5-a"

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "$SESSION is already running"
  exit 0
fi

tmux new-session -d -s "$SESSION" "cd '$PROJECT_ROOT' && exec bash scripts/v5/server_a_queue.sh"
echo "Started $SESSION. Attach with: tmux attach -t $SESSION"
