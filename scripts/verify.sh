#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export MPLCONFIGDIR=work/matplotlib
export CI=1
.venv/bin/python scripts/prepare.py data/raw/Обращения_1931.xlsx
.venv/bin/python -m pytest
.venv/bin/python -m ruff check backend/app scripts/prepare.py
.venv/bin/python -m compileall -q backend scripts
(cd frontend && pnpm install --frozen-lockfile --offline)
(cd frontend && pnpm test)
(cd frontend && pnpm typecheck)
(cd frontend && pnpm lint)
(cd frontend && pnpm build)
echo "Все автоматические проверки пройдены."
