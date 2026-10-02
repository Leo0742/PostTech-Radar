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
  --output artifacts/gpu_research_v5/environment/server_a_before.json

for stage in smoke instruction_search representation_search feature_search head_tournament repeated_eval; do
  echo "=== Server A: $stage ==="
  python scripts/v5_gpu_runner.py \
    --config configs/v5/server_a_qwen8b.json \
    --stage "$stage" \
    --batch-size "${V5_BATCH_SIZE:-8}" \
    2>&1 | tee "/workspace/logs/posttech-v5-a-$stage.log"
done

python scripts/v5/hash_stage_models.py \
  --results-root artifacts/gpu_research_v5/runs/server_a \
  --output artifacts/gpu_research_v5/environment/server_a_model_artifacts.json

python scripts/v5/capture_environment.py \
  --output artifacts/gpu_research_v5/environment/server_a_after.json
printf 'SERVER A FROZEN BASELINES COMPLETE\n' > artifacts/gpu_research_v5/runs/server_a/SERVER_A_FROZEN_BASELINES_DONE
