#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
MODEL_CACHE="$ROOT_DIR/work/model-cache-v3"
PIP_CACHE="$ROOT_DIR/work/pip-cache-v3"

if [[ -d "$MODEL_CACHE" ]]; then
  rm -rf -- "$MODEL_CACHE"
fi
if [[ -d "$PIP_CACHE" ]]; then
  rm -rf -- "$PIP_CACHE"
fi

echo "Removed isolated v3 download caches: $MODEL_CACHE and $PIP_CACHE"
