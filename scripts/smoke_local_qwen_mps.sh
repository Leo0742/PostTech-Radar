#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="$ROOT/.venv-qwen/bin/python"
URL="${POSTTECH_URL:-http://127.0.0.1:8000}"
TMP="$(mktemp)"
trap 'rm -f "$TMP"' EXIT

curl -fsS "$URL/api/health"
echo

curl -fsS -X POST "$URL/api/tickets/analyze" \
  -H 'Content-Type: application/json' \
  --data-binary '{"description":"Не могу войти в личный кабинет, после ввода пароля снова появляется форма авторизации"}' \
  -o "$TMP"

"$PYTHON" - "$TMP" <<'PY'
from __future__ import annotations

import json
import sys

result = json.load(open(sys.argv[1], encoding="utf-8"))
provenance = result.get("model_provenance", {})
print("TOP1:", result["category"]["label"], result["category"]["confidence"])
print("TOP3:", [(x["label"], x["confidence"]) for x in result["category"]["alternatives"]])
print("ROUTING:", result["routing"]["label"], result["routing"]["confidence"])
print("FALLBACK:", provenance.get("fallback"))
print("FALLBACK_REASON:", provenance.get("fallback_reason"))
print("MODEL_FAMILY:", provenance.get("model_family"))
if provenance.get("fallback"):
    raise SystemExit("Qwen smoke failed: runtime used fallback")
PY
