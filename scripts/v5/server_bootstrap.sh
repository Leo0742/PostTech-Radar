#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
WORKSPACE="${1:-$PROJECT_ROOT}"

cd "$WORKSPACE"
mkdir -p /workspace/hf-cache /workspace/checkpoints /workspace/logs
mkdir -p artifacts/gpu_research_v5/environment artifacts/gpu_research_v5/runs artifacts/gpu_research_v5/embedding_cache

if ! command -v tmux >/dev/null 2>&1 || ! python3 -c 'import venv' >/dev/null 2>&1; then
  apt-get update
  apt-get install -y tmux python3-venv
fi

if [[ ! -d .venv ]]; then
  python3 -m venv .venv
fi
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements-v5-gpu.txt

export HF_HOME="${HF_HOME:-/workspace/hf-cache}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export TOKENIZERS_PARALLELISM=false

python scripts/v5/capture_environment.py \
  --output artifacts/gpu_research_v5/environment/bootstrap_environment.json
python scripts/v5_gpu_runner.py --config configs/v5/server_a_qwen8b.json --stage smoke --dry-run >/dev/null
python scripts/v5_gpu_runner.py --config configs/v5/server_b_challengers.json --stage smoke --dry-run >/dev/null

echo "V5 server bootstrap complete in $WORKSPACE"
