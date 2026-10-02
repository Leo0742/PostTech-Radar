#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
MPLCONFIGDIR=work/matplotlib .venv/bin/python scripts/prepare.py "${1:-data/raw/Обращения_1931.xlsx}"

