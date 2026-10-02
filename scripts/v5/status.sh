#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

echo "=== tmux ==="
tmux list-sessions 2>/dev/null || true

echo "=== GPU ==="
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu,temperature.gpu --format=csv,noheader
else
  echo "nvidia-smi unavailable"
fi

echo "=== stage summaries ==="
find artifacts/gpu_research_v5/runs -name stage_summary.json -type f -print 2>/dev/null | sort | while read -r summary; do
  python - "$summary" <<'PY'
import json, sys
from pathlib import Path
path = Path(sys.argv[1])
data = json.loads(path.read_text())
print(f"{path}: completed={data.get('completed')} skipped={data.get('skipped')} failed={data.get('failed')} candidates={data.get('candidate_count')}")
PY
done

for marker in \
  artifacts/gpu_research_v5/runs/server_a/SERVER_A_FROZEN_BASELINES_DONE \
  artifacts/gpu_research_v5/runs/server_b/SERVER_B_CAN_NOW_BE_STOPPED; do
  if [[ -f "$marker" ]]; then
    cat "$marker"
  fi
done
