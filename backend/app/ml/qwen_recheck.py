from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any

from app.core.config import MODELS_DIR


class QwenRecheckError(RuntimeError):
    pass


def recheck_with_qwen(
    payload: dict[str, Any],
    *,
    timeout: int = 180,
    project_root: Path | None = None,
    models_dir: Path = MODELS_DIR,
) -> dict[str, Any]:
    root = project_root or Path(__file__).resolve().parents[3]
    qwen_python = Path(os.getenv("QWEN_WORKER_PYTHON", str(root / ".venv-qwen/bin/python")))
    local_model_dir = Path(
        os.getenv("QWEN_LOCAL_MODEL_DIR", str(root / "models/encoders/Qwen3-Embedding-4B"))
    )
    mlx_python = Path(os.getenv("QWEN_MLX_WORKER_PYTHON", str(root / ".venv-mlx/bin/python")))
    mlx_model_dir = Path(
        os.getenv(
            "QWEN_MLX_MODEL_DIR",
            str(root / "models/encoders/Qwen3-Embedding-4B-MLX-4bit"),
        )
    )
    artifact = Path(
        os.getenv(
            "QWEN_CATEGORY_ARTIFACT",
            str(models_dir / "v5" / "category_qwen4b_lite.joblib"),
        )
    )
    worker = Path(__file__).with_name("qwen_recheck_worker.py")

    requested_runtime = os.getenv("QWEN_RECHECK_RUNTIME", "auto").strip().lower() or "auto"
    if requested_runtime not in {"auto", "mlx", "torch"}:
        raise QwenRecheckError("QWEN_RECHECK_RUNTIME must be one of: auto, mlx, torch")
    mlx_ready = mlx_python.exists() and mlx_model_dir.exists()
    use_mlx = requested_runtime == "mlx" or (requested_runtime == "auto" and mlx_ready)
    worker_python = mlx_python if use_mlx else qwen_python
    required = (worker_python, mlx_model_dir if use_mlx else local_model_dir, artifact, worker)
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise QwenRecheckError("Qwen recheck is not prepared locally: " + ", ".join(missing))

    env = os.environ.copy()
    if use_mlx:
        env["QWEN_RUNTIME"] = "mlx"
        env["QWEN_MLX_MODEL_DIR"] = str(mlx_model_dir)
    else:
        env["QWEN_RUNTIME"] = "torch"
        env.setdefault("QWEN_DEVICE", "auto")
        env["QWEN_LOCAL_MODEL_DIR"] = str(local_model_dir)
        env.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    env.setdefault("TOKENIZERS_PARALLELISM", "false")

    try:
        completed = subprocess.run(
            [str(worker_python), str(worker), "--artifact", str(artifact)],
            input=json.dumps(payload, ensure_ascii=False),
            text=True,
            capture_output=True,
            timeout=timeout,
            env=env,
            cwd=str(root),
            check=False,
        )
    except subprocess.TimeoutExpired as error:
        raise QwenRecheckError(f"Qwen recheck exceeded {timeout} seconds") from error

    if completed.returncode != 0:
        message = completed.stderr.strip() or completed.stdout.strip() or "Qwen worker failed"
        raise QwenRecheckError(message[-1200:])
    try:
        return json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise QwenRecheckError("Qwen worker returned invalid JSON") from error
