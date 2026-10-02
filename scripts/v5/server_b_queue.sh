#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"
source .venv/bin/activate
export HF_HOME="${HF_HOME:-/workspace/hf-cache}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export TOKENIZERS_PARALLELISM=false
mkdir -p /workspace/logs artifacts/gpu_research_v5/environment

python scripts/v5/capture_environment.py \
  --output artifacts/gpu_research_v5/environment/server_b_before.json

for stage in smoke one_fold several_folds full_repeated; do
  echo "=== Server B: $stage ==="
  python scripts/v5_gpu_runner.py \
    --config configs/v5/server_b_challengers.json \
    --stage "$stage" \
    --batch-size "${V5_BATCH_SIZE:-8}" \
    2>&1 | tee "/workspace/logs/posttech-v5-b-$stage.log"
done

python scripts/v5/hash_stage_models.py \
  --results-root artifacts/gpu_research_v5/runs/server_b \
  --output artifacts/gpu_research_v5/environment/server_b_model_artifacts.json

python scripts/v5/capture_environment.py \
  --output artifacts/gpu_research_v5/environment/server_b_after.json
printf 'SERVER B CAN NOW BE STOPPED.\n' > artifacts/gpu_research_v5/runs/server_b/SERVER_B_CAN_NOW_BE_STOPPED
cat artifacts/gpu_research_v5/runs/server_b/SERVER_B_CAN_NOW_BE_STOPPED
