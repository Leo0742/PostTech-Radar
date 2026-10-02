from __future__ import annotations

import os
import sys
from pathlib import Path

QWEN_MODEL_ID = "Qwen/Qwen3-Embedding-4B"
QWEN_REVISION = "5cf2132abc99cad020ac570b19d031efec650f2b"

MODEL_ADDONS = (
    "models/customer_2026-09-21/category_lite_finetuned.joblib",
    "models/customer_2026-09-21/category_qwen_finetuned.joblib",
    "models/customer_2026-09-21/minilm_supcon_s120/model.safetensors",
    "models/customer_2026-09-21/qwen_r16_mnrl_s80_adapter/adapter_config.json",
    "models/customer_2026-09-21/qwen_r16_mnrl_s80_adapter/adapter_model.safetensors",
)


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def venv_python(root: Path, *, platform_name: str | None = None) -> Path:
    platform_name = platform_name or sys.platform
    if platform_name.startswith("win"):
        return root / ".venv" / "Scripts" / "python.exe"
    return root / ".venv" / "bin" / "python"


def npm_command(*, platform_name: str | None = None) -> str:
    platform_name = platform_name or sys.platform
    return "npm.cmd" if platform_name.startswith("win") else "npm"


def is_lfs_pointer(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size > 1024:
        return False
    return path.read_bytes().startswith(b"version https://git-lfs.github.com/spec/v1")


def inspect_model_addons(root: Path) -> list[str]:
    issues: list[str] = []
    for relative in MODEL_ADDONS:
        path = root / relative
        if not path.exists():
            issues.append(f"missing model add-on: {relative}")
        elif is_lfs_pointer(path):
            issues.append(f"Git LFS pointer was not downloaded: {relative}")
    return issues


def runtime_environment(root: Path, python: Path) -> dict[str, str]:
    customer = root / "models" / "customer_2026-09-21"
    return {
        "MODEL_PROFILE": "lite",
        "LITE_CATEGORY_ARTIFACT": str(customer / "category_lite_finetuned.joblib"),
        "LITE_MINILM_MODEL_DIR": str(customer / "minilm_supcon_s120"),
        "LITE_MINILM_DEVICE": "auto",
        "QWEN_CATEGORY_ARTIFACT": str(customer / "category_qwen_finetuned.joblib"),
        "QWEN_WORKER_PYTHON": str(python),
        "QWEN_LOCAL_MODEL_DIR": str(root / "models" / "encoders" / "Qwen3-Embedding-4B"),
        "QWEN_RECHECK_RUNTIME": "torch",
        "QWEN_DEVICE": "auto",
        "PYTORCH_ENABLE_MPS_FALLBACK": "1",
        "TOKENIZERS_PARALLELISM": "false",
    }


def merged_runtime_environment(root: Path, python: Path) -> dict[str, str]:
    env = os.environ.copy()
    env.update(runtime_environment(root, python))
    return env
