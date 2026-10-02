#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="$ROOT/.venv-qwen/bin/python"
MODEL_DIR="$ROOT/models/encoders/Qwen3-Embedding-4B"

if [[ ! -x "$PYTHON" ]]; then
  echo "Missing Qwen environment: $PYTHON" >&2
  exit 1
fi

for file in \
  "$MODEL_DIR/config.json" \
  "$MODEL_DIR/model-00001-of-00002.safetensors" \
  "$MODEL_DIR/model-00002-of-00002.safetensors" \
  "$ROOT/models/v5/category_qwen4b_lite.joblib"; do
  if [[ ! -f "$file" ]]; then
    echo "Missing required file: $file" >&2
    exit 1
  fi
done

export MODEL_PROFILE=lite
export QWEN_DEVICE=mps
export QWEN_LOCAL_MODEL_DIR="$MODEL_DIR"
export PYTORCH_ENABLE_MPS_FALLBACK=1
export TOKENIZERS_PARALLELISM=false

echo "PostTech Radar — local Qwen3-Embedding-4B Lite"
echo "Model: $QWEN_LOCAL_MODEL_DIR"
echo "Device: $QWEN_DEVICE"
echo "URL: http://127.0.0.1:8000"
echo "The first analysis request will load the 7.5 GB encoder into memory."

cd "$ROOT"
exec "$PYTHON" -m uvicorn app.main:app \
  --app-dir backend \
  --host 127.0.0.1 \
  --port 8000
