from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = Path(os.getenv("POSTTECH_DATA_DIR", PROJECT_ROOT / "data"))
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
DATABASE_PATH = Path(os.getenv("POSTTECH_DB_PATH", PROCESSED_DIR / "posttech.db"))
MODELS_DIR = Path(os.getenv("POSTTECH_MODELS_DIR", PROJECT_ROOT / "models"))
ARTIFACTS_DIR = Path(os.getenv("POSTTECH_ARTIFACTS_DIR", PROJECT_ROOT / "artifacts"))
EVALUATION_DIR = ARTIFACTS_DIR / "evaluation"
FIGURES_DIR = ARTIFACTS_DIR / "figures"
RANDOM_SEED = 42


def ensure_directories() -> None:
    for path in (RAW_DATA_DIR, PROCESSED_DIR, MODELS_DIR, EVALUATION_DIR, FIGURES_DIR):
        path.mkdir(parents=True, exist_ok=True)
