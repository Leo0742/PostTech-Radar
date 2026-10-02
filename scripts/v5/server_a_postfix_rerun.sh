#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

while tmux has-session -t posttech-v5-a 2>/dev/null; do
  sleep 20
done

source .venv/bin/activate
export HF_HOME="${HF_HOME:-/workspace/hf-cache}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export TOKENIZERS_PARALLELISM=false

python scripts/v5_gpu_runner.py \
  --config configs/v5/server_a_qwen8b.json \
  --stage head_tournament \
  --batch-size "${V5_BATCH_SIZE:-8}" \
  2>&1 | tee /workspace/logs/posttech-v5-a-head_tournament-fix.log

python scripts/v5_gpu_runner.py \
  --config configs/v5/server_a_qwen8b.json \
  --stage repeated_eval \
  --batch-size "${V5_BATCH_SIZE:-8}" \
  2>&1 | tee /workspace/logs/posttech-v5-a-repeated_eval-postfix.log

python scripts/v5/hash_stage_models.py \
  --results-root artifacts/gpu_research_v5/runs/server_a \
  --output artifacts/gpu_research_v5/environment/server_a_model_artifacts.json

python scripts/v5/capture_environment.py \
  --output artifacts/gpu_research_v5/environment/server_a_after_postfix.json

printf 'SERVER A POSTFIX HEAD RERUN COMPLETE\n' > \
  artifacts/gpu_research_v5/runs/server_a/SERVER_A_POSTFIX_RERUN_DONE
