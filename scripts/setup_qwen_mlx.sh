#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SOURCE="$ROOT/models/encoders/Qwen3-Embedding-4B"
TARGET="$ROOT/models/encoders/Qwen3-Embedding-4B-MLX-4bit"
VENV="$ROOT/.venv-mlx"

if [[ ! -f "$SOURCE/model.safetensors.index.json" ]]; then
  echo "Missing pinned Qwen source model: $SOURCE" >&2
  exit 1
fi

if [[ ! -x "$VENV/bin/python" ]]; then
  python3 -m venv "$VENV"
fi

"$VENV/bin/pip" install --upgrade pip
"$VENV/bin/pip" install \
  "mlx-embeddings==0.1.0" \
  "scikit-learn==1.8.0" \
  "joblib==1.5.2" \
  "scipy==1.18.1"

if [[ ! -f "$TARGET/model.safetensors" ]]; then
  "$VENV/bin/python" -m mlx_embeddings.convert \
    --hf-path "$SOURCE" \
    --mlx-path "$TARGET" \
    --quantize \
    --q-group-size 64 \
    --q-bits 4
fi

echo "Qwen MLX runtime is ready: $TARGET"
