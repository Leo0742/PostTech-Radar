from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
V5_DIR = PROJECT_ROOT / "scripts" / "v5"


def test_v5_gpu_requirements_pin_direct_research_dependencies() -> None:
    text = (PROJECT_ROOT / "requirements-v5-gpu.txt").read_text(encoding="utf-8")

    for pin in (
        "catboost==1.2.10",
        "transformers==4.57.6",
        "sentence-transformers==5.1.2",
        "accelerate==1.10.1",
        "peft==0.17.1",
        "safetensors==0.6.2",
        "scipy==1.16.3",
    ):
        assert pin in text
    assert "torch==2.8.0" in text
    assert "cu128" in text


def test_server_bootstrap_prepares_reproducible_workspace_and_tmux() -> None:
    text = (V5_DIR / "server_bootstrap.sh").read_text(encoding="utf-8")

    assert "python3 -m venv" in text
    assert "requirements-v5-gpu.txt" in text
    assert "/workspace/hf-cache" in text
    assert "/workspace/checkpoints" in text
    assert "/workspace/logs" in text
    assert "tmux" in text
    assert "capture_environment.py" in text


def test_environment_capture_and_gpu_smoke_are_directly_invokable() -> None:
    for script in ("capture_environment.py", "smoke_gpu.py", "hash_stage_models.py"):
        result = subprocess.run(
            [sys.executable, str(V5_DIR / script), "--help"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr


def test_launch_scripts_use_tmux_and_expected_stage_queues() -> None:
    server_a = (V5_DIR / "run_server_a.sh").read_text(encoding="utf-8")
    server_b = (V5_DIR / "run_server_b.sh").read_text(encoding="utf-8")
    queue_a = (V5_DIR / "server_a_queue.sh").read_text(encoding="utf-8")
    queue_b = (V5_DIR / "server_b_queue.sh").read_text(encoding="utf-8")

    assert "posttech-v5-a" in server_a
    assert "posttech-v5-b" in server_b
    assert "tmux new-session" in server_a
    assert "tmux new-session" in server_b
    for stage in ("smoke", "instruction_search", "representation_search", "feature_search", "head_tournament", "repeated_eval"):
        assert stage in queue_a
    for stage in ("smoke", "one_fold", "several_folds", "full_repeated"):
        assert stage in queue_b
    assert "hash_stage_models.py" in queue_a
    assert "hash_stage_models.py" in queue_b
    assert "SERVER B CAN NOW BE STOPPED" in queue_b


def test_status_script_reports_tmux_gpu_and_stage_summaries() -> None:
    text = (V5_DIR / "status.sh").read_text(encoding="utf-8")

    assert "tmux" in text
    assert "nvidia-smi" in text
    assert "stage_summary.json" in text


def test_result_schema_requires_core_research_provenance() -> None:
    schema = json.loads((PROJECT_ROOT / "configs" / "v5" / "result_schema.json").read_text())

    assert set(schema["required"]) >= {"status", "candidate", "metrics", "provenance", "runtime"}
    assert set(schema["properties"]["metrics"]["required"]) >= {
        "top1_accuracy",
        "top3_accuracy",
        "macro_f1",
        "true_label_mrr",
    }


def test_all_v5_shell_assets_are_syntax_valid() -> None:
    scripts = sorted(V5_DIR.glob("*.sh"))
    assert scripts
    for script in scripts:
        result = subprocess.run(["bash", "-n", str(script)], capture_output=True, text=True, check=False)
        assert result.returncode == 0, f"{script}: {result.stderr}"
