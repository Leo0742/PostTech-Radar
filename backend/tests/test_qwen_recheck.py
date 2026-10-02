from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from app.ml.qwen_recheck import QwenRecheckError, recheck_with_qwen


def test_recheck_runs_isolated_qwen_worker_and_parses_result(tmp_path, monkeypatch) -> None:
    project_root = tmp_path
    worker_python = project_root / ".venv-qwen" / "bin" / "python"
    worker_python.parent.mkdir(parents=True)
    worker_python.write_text("", encoding="utf-8")
    local_model = project_root / "models" / "encoders" / "Qwen3-Embedding-4B"
    local_model.mkdir(parents=True)
    models_dir = project_root / "models"
    (models_dir / "v5").mkdir(parents=True)
    (models_dir / "v5" / "category_qwen4b_lite.joblib").write_bytes(b"stub")
    captured = {}

    def fake_run(command, **kwargs):
        captured["command"] = command
        captured.update(kwargs)
        return SimpleNamespace(
            returncode=0,
            stdout=json.dumps({
                "category": {
                    "label": "Личный кабинет",
                    "confidence": 0.73,
                    "alternatives": [
                        {"label": "Личный кабинет", "confidence": 0.73},
                        {"label": "Недоступность портала", "confidence": 0.19},
                    ],
                },
                "model_provenance": {"model_id": "Qwen/Qwen3-Embedding-4B"},
                "latency_seconds": 12.3,
            }, ensure_ascii=False),
            stderr="",
        )

    monkeypatch.setattr("app.ml.qwen_recheck.subprocess.run", fake_run)

    result = recheck_with_qwen(
        {"description": "не могу войти в личный кабинет"},
        timeout=90,
        project_root=project_root,
        models_dir=models_dir,
    )

    assert result["category"]["label"] == "Личный кабинет"
    assert captured["command"][0] == str(worker_python)
    assert captured["env"]["QWEN_DEVICE"] == "auto"
    assert captured["input"] == json.dumps(
        {"description": "не могу войти в личный кабинет"}, ensure_ascii=False
    )
    assert captured["timeout"] == 90


def test_recheck_prefers_ephemeral_mlx_worker_when_local_runtime_is_ready(tmp_path, monkeypatch) -> None:
    project_root = tmp_path
    qwen_python = project_root / ".venv-qwen" / "bin" / "python"
    qwen_python.parent.mkdir(parents=True)
    qwen_python.write_text("", encoding="utf-8")
    torch_model = project_root / "models" / "encoders" / "Qwen3-Embedding-4B"
    torch_model.mkdir(parents=True)
    mlx_python = project_root / ".venv-mlx" / "bin" / "python"
    mlx_python.parent.mkdir(parents=True)
    mlx_python.write_text("", encoding="utf-8")
    mlx_model = project_root / "models" / "encoders" / "Qwen3-Embedding-4B-MLX-4bit"
    mlx_model.mkdir(parents=True)
    models_dir = project_root / "models"
    (models_dir / "v5").mkdir(parents=True)
    (models_dir / "v5" / "category_qwen4b_lite.joblib").write_bytes(b"stub")
    captured = {}

    def fake_run(command, **kwargs):
        captured["command"] = command
        captured.update(kwargs)
        return SimpleNamespace(
            returncode=0,
            stdout=json.dumps({"category": {"label": "A", "alternatives": []}, "latency_seconds": 2.0}),
            stderr="",
        )

    monkeypatch.setattr("app.ml.qwen_recheck.subprocess.run", fake_run)

    recheck_with_qwen(
        {"description": "test"},
        project_root=project_root,
        models_dir=models_dir,
    )

    assert captured["command"][0] == str(mlx_python)
    assert captured["env"]["QWEN_RUNTIME"] == "mlx"
    assert captured["env"]["QWEN_MLX_MODEL_DIR"] == str(mlx_model)
    assert captured["timeout"] == 180


def test_recheck_allows_explicit_mlx_runtime_with_category_artifact(tmp_path, monkeypatch) -> None:
    project_root = tmp_path
    qwen_python = project_root / ".venv-qwen" / "bin" / "python"
    qwen_python.parent.mkdir(parents=True)
    qwen_python.write_text("", encoding="utf-8")
    (project_root / "models" / "encoders" / "Qwen3-Embedding-4B").mkdir(parents=True)
    mlx_python = project_root / ".venv-mlx" / "bin" / "python"
    mlx_python.parent.mkdir(parents=True)
    mlx_python.write_text("", encoding="utf-8")
    mlx_model = project_root / "models" / "encoders" / "Qwen3-Embedding-4B-MLX-4bit"
    mlx_model.mkdir(parents=True)
    models_dir = project_root / "models"
    (models_dir / "v5").mkdir(parents=True)
    artifact = models_dir / "v5" / "category_qwen4b_lite.joblib"
    artifact.write_bytes(b"stub")
    captured = {}

    monkeypatch.setenv("QWEN_RECHECK_RUNTIME", "mlx")

    def fake_run(command, **kwargs):
        captured["command"] = command
        captured.update(kwargs)
        return SimpleNamespace(
            returncode=0,
            stdout=json.dumps({"category": {"label": "A", "alternatives": []}}),
            stderr="",
        )

    monkeypatch.setattr("app.ml.qwen_recheck.subprocess.run", fake_run)
    recheck_with_qwen(
        {"description": "test"},
        project_root=project_root,
        models_dir=models_dir,
    )

    assert captured["command"][0] == str(mlx_python)
    assert captured["env"]["QWEN_RUNTIME"] == "mlx"


def test_recheck_surfaces_worker_failure(tmp_path, monkeypatch) -> None:
    worker_python = tmp_path / ".venv-qwen" / "bin" / "python"
    worker_python.parent.mkdir(parents=True)
    worker_python.write_text("", encoding="utf-8")
    local_model = tmp_path / "models" / "encoders" / "Qwen3-Embedding-4B"
    local_model.mkdir(parents=True)
    models_dir = tmp_path / "models"
    (models_dir / "v5").mkdir(parents=True)
    (models_dir / "v5" / "category_qwen4b_lite.joblib").write_bytes(b"stub")

    monkeypatch.setattr(
        "app.ml.qwen_recheck.subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(returncode=1, stdout="", stderr="OOM while loading Qwen"),
    )

    with pytest.raises(QwenRecheckError, match="OOM while loading Qwen"):
        recheck_with_qwen(
            {"description": "test"},
            project_root=tmp_path,
            models_dir=models_dir,
        )
