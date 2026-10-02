#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

# Never compete with the current Qwen8B queue or its already-queued postfix rerun.
while tmux has-session -t posttech-v5-a 2>/dev/null || tmux has-session -t posttech-v5-postfix 2>/dev/null; do
  sleep 20
done

test -f artifacts/gpu_research_v5/runs/server_a/SERVER_A_FROZEN_BASELINES_DONE
test -f artifacts/gpu_research_v5/runs/server_a/SERVER_A_POSTFIX_RERUN_DONE

source .venv/bin/activate
export HF_HOME=/workspace/hf-cache
export HF_HUB_CACHE=/workspace/hf-cache/hub
export TOKENIZERS_PARALLELISM=false
mkdir -p /workspace/logs artifacts/gpu_research_v5/matched_reference

python scripts/v5/prepare_matched_qwen_reference.py \
  2>&1 | tee /workspace/logs/posttech-v5-matched-prepare.log

# Evaluate both models at the exact same comparable recipe.
python scripts/v5_gpu_runner.py \
  --config configs/v5/matched_qwen8b.json \
  --stage matched_eval \
  --batch-size 8 \
  2>&1 | tee /workspace/logs/posttech-v5-matched-qwen8b.log

python scripts/v5_gpu_runner.py \
  --config configs/v5/matched_qwen4b.json \
  --stage matched_eval \
  --batch-size 8 \
  2>&1 | tee /workspace/logs/posttech-v5-matched-qwen4b.log

# Fresh-cache one-fold benchmarks provide comparable latency and peak VRAM.
STAMP="$(date +%Y%m%d_%H%M%S)"
python scripts/v5_gpu_runner.py \
  --config configs/v5/matched_benchmark_qwen8b.json \
  --stage matched_benchmark \
  --batch-size 8 \
  --cache-root "artifacts/gpu_research_v5/matched_reference/cache_qwen8b_$STAMP" \
  2>&1 | tee /workspace/logs/posttech-v5-matched-benchmark-qwen8b.log

python scripts/v5_gpu_runner.py \
  --config configs/v5/matched_benchmark_qwen4b.json \
  --stage matched_benchmark \
  --batch-size 8 \
  --cache-root "artifacts/gpu_research_v5/matched_reference/cache_qwen4b_$STAMP" \
  2>&1 | tee /workspace/logs/posttech-v5-matched-benchmark-qwen4b.log

python scripts/v5/report_matched_qwen_reference.py \
  2>&1 | tee /workspace/logs/posttech-v5-matched-report.log

printf 'MATCHED QWEN4B VS QWEN8B V5 REFERENCE COMPLETE\n' > \
  artifacts/gpu_research_v5/matched_reference/MATCHED_REFERENCE_DONE
