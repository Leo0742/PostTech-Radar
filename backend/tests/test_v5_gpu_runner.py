from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from scripts.v5_experiment import load_json
from scripts.v5_gpu_runner import (
    DiskEmbeddingCache,
    build_structured_features,
    evaluate_candidate_with_embedder,
    execute_stage,
    load_completed_stage_results,
    output_dir_for_run,
    pin_candidate_revision,
    resolve_stage_candidates,
    select_promoted_candidates,
    should_skip_run,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = PROJECT_ROOT / "artifacts" / "gpu_research_v5" / "protocol" / "protocol.json"
SERVER_A_CONFIG = PROJECT_ROOT / "configs" / "v5" / "server_a_qwen8b.json"


def _result(candidate: dict, *, top1: float, macro: float) -> dict:
    return {
        "status": "complete",
        "candidate": candidate,
        "metrics": {
            "top1_accuracy": top1,
            "macro_f1": macro,
            "top3_accuracy": 0.95,
            "true_label_mrr": 0.85,
        },
    }


def test_promotion_aggregates_fold_results_and_keeps_quality_first_winner() -> None:
    a0 = {"model_id": "A", "instruction": "none", "repeat": 0, "fold": 0, "run_id": "a0", "stage": "s"}
    a1 = {"model_id": "A", "instruction": "none", "repeat": 0, "fold": 1, "run_id": "a1", "stage": "s"}
    b0 = {"model_id": "B", "instruction": "none", "repeat": 0, "fold": 0, "run_id": "b0", "stage": "s"}
    b1 = {"model_id": "B", "instruction": "none", "repeat": 0, "fold": 1, "run_id": "b1", "stage": "s"}

    promoted = select_promoted_candidates(
        [
            _result(a0, top1=0.80, macro=0.60),
            _result(a1, top1=0.82, macro=0.60),
            _result(b0, top1=0.79, macro=0.99),
            _result(b1, top1=0.80, macro=0.99),
        ],
        top_n=1,
    )

    assert len(promoted) == 1
    assert promoted[0]["candidate"]["model_id"] == "A"
    assert promoted[0]["aggregate_metrics"]["top1_accuracy"] == 0.81


def test_result_discovery_only_returns_completed_json(tmp_path: Path) -> None:
    complete = tmp_path / "source" / "run-a"
    failed = tmp_path / "source" / "run-b"
    complete.mkdir(parents=True)
    failed.mkdir(parents=True)
    (complete / "result.json").write_text(json.dumps(_result({"run_id": "a", "stage": "source"}, top1=0.8, macro=0.7)))
    (failed / "result.json").write_text(json.dumps({"status": "failed", "candidate": {"run_id": "b"}}))

    found = load_completed_stage_results(tmp_path, "source")

    assert [item["candidate"]["run_id"] for item in found] == ["a"]


def test_output_directory_and_resume_skip_are_deterministic(tmp_path: Path) -> None:
    candidate = {"stage": "smoke", "run_id": "smoke-123"}
    output = output_dir_for_run(tmp_path, candidate)
    output.mkdir(parents=True)

    assert output == tmp_path / "smoke" / "smoke-123"
    assert should_skip_run(output) is False
    (output / "result.json").write_text(json.dumps({"status": "complete"}))
    assert should_skip_run(output) is True


def test_inherited_stage_reuses_best_instruction(tmp_path: Path) -> None:
    protocol = load_json(PROTOCOL)
    config = load_json(SERVER_A_CONFIG)
    source = tmp_path / "instruction_search" / "winner"
    source.mkdir(parents=True)
    winner = {
        "stage": "instruction_search",
        "server": "A",
        "model_id": "Qwen/Qwen3-Embedding-8B",
        "model_revision": "abc123",
        "deployment_policy": "normal",
        "view": "top15",
        "feature_mode": "text_only",
        "instruction": "ru_concise",
        "embedding_dim": 2048,
        "max_length": 256,
        "head": "logreg",
        "repeat": 0,
        "fold": 0,
        "seed": 20260917,
        "run_id": "winner",
    }
    (source / "result.json").write_text(json.dumps(_result(winner, top1=0.85, macro=0.8)))

    candidates = resolve_stage_candidates(config, "representation_search", protocol=protocol, results_root=tmp_path)

    assert candidates
    assert {item["instruction"] for item in candidates} == {"ru_concise"}
    assert {item["model_revision"] for item in candidates} == {"abc123"}
    assert len(candidates) == 24


def test_dry_run_cli_does_not_require_gpu_dependencies() -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "v5_gpu_runner.py"),
            "--config",
            str(SERVER_A_CONFIG),
            "--stage",
            "smoke",
            "--dry-run",
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["stage"] == "smoke"
    assert payload["candidate_count"] == 1
    assert payload["candidates"][0]["model_id"] == "Qwen/Qwen3-Embedding-8B"


def test_structured_features_are_missing_aware_and_derive_registration_time() -> None:
    rows = [
        {
            "registration_date": "2026-09-17T13:45:00.000000",
            "user": "u1",
            "service": "svc",
            "component": None,
            "request_type": "incident",
            "criticality": "high",
            "urgency": None,
            "priority": "p1",
            "service_class": "gold",
            "timezone": "+03:00",
        }
    ]

    matrix = build_structured_features(rows)

    assert matrix.shape == (1, 12)
    assert "__MISSING__" in matrix[0].tolist()
    assert "None" not in matrix[0].tolist()
    assert matrix[0, -3:].tolist() == ["9", "3", "13"]


def test_candidate_evaluation_core_works_with_injected_cpu_embedder() -> None:
    rows = []
    for index in range(8):
        label = "A" if index < 4 else "B"
        rows.append(
            {
                "request_id": str(index),
                "category": label,
                "routing_target": "(1 линия)",
                "registration_date": f"2026-09-{10 + index:02d}T10:00:00.000000",
                "user": "u",
                "service": "svc-a" if label == "A" else "svc-b",
                "component": None,
                "request_type": "incident",
                "description": "apple ticket" if label == "A" else "banana ticket",
                "criticality": "normal",
                "urgency": "normal",
                "priority": "p2",
                "service_class": "std",
                "timezone": "+03:00",
            }
        )

    train_ids = ["0", "1", "2", "4", "5", "6"]
    validation_ids = ["3", "7"]
    candidate = {
        "stage": "unit",
        "server": "local",
        "model_id": "dummy",
        "model_revision": "dummy",
        "deployment_policy": "normal",
        "view": "top15",
        "feature_mode": "text_only",
        "instruction": "none",
        "embedding_dim": 2,
        "max_length": 16,
        "head": "logreg",
        "repeat": 0,
        "fold": 0,
        "seed": 1,
        "run_id": "unit-1",
    }

    def embedder(texts, _candidate):
        result = []
        for text in texts:
            lower = text.lower()
            if "apple" in lower:
                result.append([1.0, 0.0])
            elif "banana" in lower:
                result.append([0.0, 1.0])
            elif text == "A":
                result.append([1.0, 0.0])
            elif text == "B":
                result.append([0.0, 1.0])
            else:
                result.append([0.5, 0.5])
        return np.asarray(result, dtype=float)

    result = evaluate_candidate_with_embedder(
        candidate,
        rows,
        train_ids=train_ids,
        validation_ids=validation_ids,
        labels=["A", "B"],
        embedder=embedder,
    )

    assert result["status"] == "complete"
    assert result["metrics"]["top1_accuracy"] == 1.0
    assert result["metrics"]["top3_accuracy"] == 1.0
    assert result["partition"]["train_rows"] == 6
    assert result["partition"]["validation_rows"] == 2
    assert result["internal_lockbox_accessed"] is False


def test_metadata_only_fallback_uses_structured_branch_without_embeddings() -> None:
    rows = []
    for index in range(8):
        label = "A" if index < 4 else "B"
        rows.append(
            {
                "request_id": str(index),
                "category": label,
                "routing_target": "(1 линия)",
                "registration_date": f"2026-09-{10 + index:02d}T10:00:00.000000",
                "user": "u",
                "service": "svc-a" if label == "A" else "svc-b",
                "component": None,
                "request_type": "incident",
                "description": None,
                "criticality": "normal",
                "urgency": "normal",
                "priority": "p2",
                "service_class": "std",
                "timezone": "+03:00",
            }
        )
    candidate = {
        "stage": "unit",
        "server": "local",
        "model_id": "dummy",
        "model_revision": "dummy",
        "deployment_policy": "normal",
        "view": "top15",
        "feature_mode": "metadata_only_fallback",
        "instruction": "none",
        "embedding_dim": 2,
        "max_length": 16,
        "head": "logreg",
        "repeat": 0,
        "fold": 0,
        "seed": 1,
        "run_id": "unit-meta",
    }

    def forbidden_embedder(_texts, _candidate):
        raise AssertionError("metadata-only fallback must not call the embedding model")

    result = evaluate_candidate_with_embedder(
        candidate,
        rows,
        train_ids=["0", "1", "2", "4", "5", "6"],
        validation_ids=["3", "7"],
        labels=["A", "B"],
        embedder=forbidden_embedder,
    )

    assert result["status"] == "complete"
    assert result["metrics"]["top1_accuracy"] == 1.0
    assert result["effective_head"] == "metadata_logreg_fallback"


def test_embedding_cache_reuses_same_representation_across_heads_and_folds(tmp_path: Path) -> None:
    calls: list[list[str]] = []

    def backend(texts, _candidate):
        calls.append(list(texts))
        return np.arange(len(texts) * 4, dtype=float).reshape(len(texts), 4)

    cache = DiskEmbeddingCache(tmp_path, backend)
    candidate = {
        "model_id": "dummy",
        "model_revision": "abc",
        "feature_mode": "text_only",
        "instruction": "none",
        "max_length": 256,
        "embedding_dim": 4,
        "head": "logreg",
        "repeat": 0,
        "fold": 0,
    }
    first = cache(["one", "two"], candidate)
    second_candidate = dict(candidate, head="knn", repeat=1, fold=3)
    second = cache(["one", "two"], second_candidate)

    assert len(calls) == 1
    assert np.array_equal(first, second)
    assert cache.hits == 1
    assert cache.misses == 1


def test_revision_pinning_turns_mutable_model_reference_into_immutable_sha() -> None:
    candidate = {
        "stage": "smoke",
        "model_id": "org/model",
        "model_revision": None,
        "run_id": "old",
    }

    pinned, metadata = pin_candidate_revision(
        candidate,
        resolver=lambda model_id: {"model_id": model_id, "revision": "deadbeef", "license": "apache-2.0"},
    )

    assert pinned["model_revision"] == "deadbeef"
    assert pinned["run_id"] != "old"
    assert metadata["license"] == "apache-2.0"


def test_execute_stage_writes_checkpointed_result_and_resumes_without_recompute(tmp_path: Path) -> None:
    rows = []
    for index in range(8):
        label = "A" if index < 4 else "B"
        rows.append(
            {
                "request_id": str(index),
                "category": label,
                "routing_target": "(1 линия)",
                "registration_date": f"2026-09-{10 + index:02d}T10:00:00.000000",
                "user": "u",
                "service": "svc-a" if label == "A" else "svc-b",
                "component": None,
                "request_type": "incident",
                "description": "apple ticket" if label == "A" else "banana ticket",
                "criticality": "normal",
                "urgency": "normal",
                "priority": "p2",
                "service_class": "std",
                "timezone": "+03:00",
            }
        )
    protocol = {
        "dataset_sha256": "dataset",
        "split_sha256": "split",
        "internal_lockbox_request_ids": ["99"],
        "folds": [
            {
                "repeat": 0,
                "fold": 0,
                "seed": 1,
                "train_request_ids": ["0", "1", "2", "4", "5", "6"],
                "validation_request_ids": ["3", "7"],
            }
        ],
    }
    contract = {
        "dataset": {"sha256": "source"},
        "category_summary": {
            "top15": [{"name": "A"}, {"name": "B"}],
            "all_categories": [{"name": "A"}, {"name": "B"}],
        },
    }
    config = {
        "server": "local",
        "artifact_root": str(tmp_path / "runs"),
        "model": {"id": "dummy", "revision": None, "deployment_policy": "normal"},
        "stages": [
            {
                "name": "smoke",
                "views": ["top15"],
                "feature_modes": ["text_only"],
                "instructions": ["none"],
                "embedding_dims": [2],
                "max_lengths": [16],
                "heads": ["logreg"],
                "repeats": [0],
                "folds": [0],
            }
        ],
    }
    backend_calls = 0

    def backend(texts, _candidate):
        nonlocal backend_calls
        backend_calls += 1
        values = []
        for text in texts:
            values.append([1.0, 0.0] if "apple" in text else [0.0, 1.0])
        return np.asarray(values, dtype=float)

    def resolver(model_id: str) -> dict[str, str]:
        return {"model_id": model_id, "revision": "sha123", "license": "test"}

    first = execute_stage(
        config,
        "smoke",
        protocol=protocol,
        contract=contract,
        rows=rows,
        results_root=tmp_path / "runs",
        cache_root=tmp_path / "cache",
        resolver=resolver,
        backend=backend,
    )
    calls_after_first = backend_calls
    second = execute_stage(
        config,
        "smoke",
        protocol=protocol,
        contract=contract,
        rows=rows,
        results_root=tmp_path / "runs",
        cache_root=tmp_path / "cache",
        resolver=resolver,
        backend=backend,
    )

    assert first["completed"] == 1
    assert first["failed"] == 0
    assert second["skipped"] == 1
    assert backend_calls == calls_after_first
    result_files = list((tmp_path / "runs" / "smoke").glob("*/result.json"))
    assert len(result_files) == 1
    payload = json.loads(result_files[0].read_text())
    assert payload["status"] == "complete"
    assert payload["candidate"]["model_revision"] == "sha123"
    assert payload["provenance"]["split_sha256"] == "split"
