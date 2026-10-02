#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
QWEN_PYTHON="$ROOT/.venv-qwen/bin/python"
QWEN_MLX_PYTHON="$ROOT/.venv-mlx/bin/python"
PYTHON="$QWEN_PYTHON"
LITE_ARTIFACT="$ROOT/models/customer_2026-09-21/category_lite_finetuned.joblib"
LITE_MODEL_DIR="$ROOT/models/customer_2026-09-21/minilm_supcon_s120"
QWEN_ARTIFACT="$ROOT/models/customer_2026-09-21/category_qwen_finetuned.joblib"
QWEN_MODEL_DIR="$ROOT/models/encoders/Qwen3-Embedding-4B"
QWEN_MLX_MODEL_DIR="$ROOT/models/encoders/Qwen3-Embedding-4B-MLX-4bit"

for file in "$PYTHON" "$LITE_ARTIFACT" "$LITE_MODEL_DIR/config.json" "$QWEN_ARTIFACT" "$QWEN_MODEL_DIR/config.json"; do
  if [[ ! -e "$file" ]]; then
    echo "Missing required local file: $file" >&2
    exit 1
  fi
done

export MODEL_PROFILE=lite
export LITE_CATEGORY_ARTIFACT="$LITE_ARTIFACT"
export LITE_MINILM_MODEL_DIR="$LITE_MODEL_DIR"
export LITE_MINILM_DEVICE=auto
export QWEN_CATEGORY_ARTIFACT="$QWEN_ARTIFACT"
export QWEN_WORKER_PYTHON="$QWEN_PYTHON"
export QWEN_LOCAL_MODEL_DIR="$QWEN_MODEL_DIR"
export QWEN_MLX_WORKER_PYTHON="$QWEN_MLX_PYTHON"
export QWEN_MLX_MODEL_DIR="$QWEN_MLX_MODEL_DIR"
export QWEN_RECHECK_RUNTIME=auto
export QWEN_DEVICE=mps
export PYTORCH_ENABLE_MPS_FALLBACK=1
export TOKENIZERS_PARALLELISM=false

echo "PostTech Radar — hybrid local mode"
echo "Default category model: Lite V3 distilled (MiniLM + Qwen4B teacher projection)"
echo "Qwen4B: fine-tuned PEFT model on 4-bit MLX, isolated on-demand; process exits after every request"
echo "URL: http://127.0.0.1:8000"

cd "$ROOT"
exec "$PYTHON" -m uvicorn app.main:app \
  --app-dir backend \
  --host 127.0.0.1 \
  --port 8000
