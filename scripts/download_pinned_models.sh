#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
PROFILE="${1:-${MODEL_PROFILE:-lite}}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python environment not found at $PYTHON_BIN" >&2
  echo "Create .venv and install requirements-v5-gpu.txt first." >&2
  exit 1
fi

download_model() {
  local model_id="$1"
  local model_revision="$2"
  "$PYTHON_BIN" - "$model_id" "$model_revision" <<'PY'
from __future__ import annotations

import sys

from huggingface_hub import snapshot_download

model_id = sys.argv[1]
revision = sys.argv[2]
path = snapshot_download(repo_id=model_id, revision=revision)
print(f"Pinned model ready: {model_id}")
print(f"Revision: {revision}")
print(f"Cache path: {path}")
PY
}

case "$PROFILE" in
  quality)
    download_model "Qwen/Qwen3-Embedding-8B" "1d8ad4ca9b3dd8059ad90a75d4983776a23d44af"
    ;;
  lite)
    download_model "Qwen/Qwen3-Embedding-4B" "5cf2132abc99cad020ac570b19d031efec650f2b"
    ;;
  all)
    download_model "Qwen/Qwen3-Embedding-8B" "1d8ad4ca9b3dd8059ad90a75d4983776a23d44af"
    download_model "Qwen/Qwen3-Embedding-4B" "5cf2132abc99cad020ac570b19d031efec650f2b"
    ;;
  *)
    echo "Unknown model profile: $PROFILE (expected quality, lite, or all)" >&2
    exit 2
    ;;
esac
