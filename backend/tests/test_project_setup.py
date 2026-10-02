from __future__ import annotations

from pathlib import Path

from scripts.audit_submission import included
from scripts.project_setup import (
    QWEN_REVISION,
    inspect_model_addons,
    runtime_environment,
    venv_python,
)


def test_venv_python_uses_platform_specific_layout(tmp_path: Path) -> None:
    assert venv_python(tmp_path, platform_name="win32") == tmp_path / ".venv" / "Scripts" / "python.exe"
    assert venv_python(tmp_path, platform_name="linux") == tmp_path / ".venv" / "bin" / "python"
    assert venv_python(tmp_path, platform_name="darwin") == tmp_path / ".venv" / "bin" / "python"


def test_inspect_model_addons_reports_missing_and_lfs_pointer_files(tmp_path: Path) -> None:
    adapter = (
        tmp_path
        / "models"
        / "customer_2026-09-21"
        / "qwen_r16_mnrl_s80_adapter"
        / "adapter_model.safetensors"
    )
    adapter.parent.mkdir(parents=True)
    adapter.write_text(
        "version https://git-lfs.github.com/spec/v1\n"
        "oid sha256:0123456789\n"
        "size 47000000\n",
        encoding="utf-8",
    )

    issues = inspect_model_addons(tmp_path)

    assert any("Git LFS pointer" in issue and "adapter_model.safetensors" in issue for issue in issues)
    assert any("missing" in issue and "minilm_supcon_s120/model.safetensors" in issue for issue in issues)


def test_runtime_environment_selects_final_finetuned_artifacts(tmp_path: Path) -> None:
    python = venv_python(tmp_path, platform_name="linux")

    env = runtime_environment(tmp_path, python)

    assert env["MODEL_PROFILE"] == "lite"
    assert env["LITE_CATEGORY_ARTIFACT"].endswith("category_lite_finetuned.joblib")
    assert env["LITE_MINILM_MODEL_DIR"].endswith("minilm_supcon_s120")
    assert env["QWEN_CATEGORY_ARTIFACT"].endswith("category_qwen_finetuned.joblib")
    assert env["QWEN_LOCAL_MODEL_DIR"].endswith("models/encoders/Qwen3-Embedding-4B")
    assert env["QWEN_WORKER_PYTHON"] == str(python)
    assert env["QWEN_DEVICE"] == "auto"
    assert len(QWEN_REVISION) == 40


def test_submission_keeps_runtime_v4_module_but_excludes_old_research_script(tmp_path: Path) -> None:
    runtime = tmp_path / "backend" / "app" / "ml" / "v4" / "runtime.py"
    research = tmp_path / "scripts" / "v4" / "benchmark.py"
    runtime.parent.mkdir(parents=True)
    research.parent.mkdir(parents=True)
    runtime.write_text("", encoding="utf-8")
    research.write_text("", encoding="utf-8")

    assert included(runtime, tmp_path) is True
    assert included(research, tmp_path) is False
