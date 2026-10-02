"""Integration contract for the Qwen3-Embedding-4B Lite default profile.

These tests never download the 4B encoder and never require CUDA: runtime
selection, fallback order and the public response schema are exercised with
tiny serialized stubs, exactly like deployment smoke tests do.
"""

from __future__ import annotations

import sqlite3

import joblib
import numpy as np
import pytest
from app.ml import v5_runtime
from app.ml.runtime import analyze_ticket, load_runtime
from app.ml.v5_runtime import QwenV5CategoryPipeline


class _MetadataEncoder:
    def transform(self, values):
        return np.zeros((len(values), 1), dtype=float)


class _ConstantMetadataEncoder:
    def __init__(self, value: float = 4.0) -> None:
        self.value = value

    def transform(self, values):
        return np.full((len(values), 1), self.value, dtype=float)


class _ProbabilityModel:
    classes_ = np.asarray(["A", "B"])

    def predict_proba(self, values):
        return np.tile(np.asarray([[0.8, 0.2]], dtype=float), (len(values), 1))


class _RecordingProbabilityModel(_ProbabilityModel):
    def __init__(self) -> None:
        self.last_values: np.ndarray | None = None

    def predict_proba(self, values):
        self.last_values = np.asarray(values, dtype=float)
        return super().predict_proba(values)


class _KnnModel:
    classes_ = np.asarray(["A", "B"])

    def predict_proba(self, values):
        return np.tile(np.asarray([[0.6, 0.4]], dtype=float), (len(values), 1))


def _pipeline(**overrides) -> QwenV5CategoryPipeline:
    params = {
        "labels": ["A", "B"],
        "model_id": "local/test",
        "model_revision": "abc",
        "instruction": "",
        "max_length": 32,
        "embedding_dim": 2,
        "metadata_encoder": _MetadataEncoder(),
        "supervised_model": _ProbabilityModel(),
        "prototype_centroids": np.asarray([[1.0, 0.0], [0.0, 1.0]]),
        "knn_model": _KnnModel(),
        "batch_size": 8,
    }
    params.update(overrides)
    pipeline = QwenV5CategoryPipeline(**params)
    pipeline._embed = lambda rows: np.tile(np.asarray([[1.0, 0.0]]), (len(rows), 1))  # type: ignore[method-assign]
    return pipeline


def _write_stub_models(tmp_path, *, lite: bool = True, corrupted_lite: bool = False) -> None:
    joblib.dump({"name": "legacy", "top15": ["A"], "threshold": 0.5}, tmp_path / "category.joblib")
    joblib.dump({"name": "routing"}, tmp_path / "routing.joblib")
    joblib.dump({"name": "retrieval"}, tmp_path / "retrieval.joblib")
    (tmp_path / "v5").mkdir(exist_ok=True)
    joblib.dump({"name": "quality", "top15": ["A"], "threshold": 0.5}, tmp_path / "v5" / "category.joblib")
    joblib.dump({"name": "qwen4b", "top15": ["A"], "threshold": 0.5}, tmp_path / "v5" / "category_qwen4b_lite.joblib")
    if lite:
        lite_path = tmp_path / "v5" / "category_lite_v2.joblib"
        if corrupted_lite:
            lite_path.write_bytes(b"not a joblib artifact")
        else:
            joblib.dump({"name": "lite", "top15": ["A"], "threshold": 0.5}, lite_path)


def test_default_profile_is_lite_when_env_is_unset(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("MODEL_PROFILE", raising=False)
    _write_stub_models(tmp_path)

    loaded = load_runtime(tmp_path)

    assert loaded["model_profile_requested"] == "lite"
    assert loaded["model_profile_active"] == "lite"
    assert loaded["category"]["name"] == "lite"
    assert loaded["category_artifact"].endswith("category_lite_v2.joblib")


def test_env_variable_switches_between_lite_and_quality(tmp_path, monkeypatch) -> None:
    _write_stub_models(tmp_path)

    monkeypatch.setenv("MODEL_PROFILE", "quality")
    quality = load_runtime(tmp_path)
    assert quality["model_profile_active"] == "quality"
    assert quality["category"]["name"] == "quality"
    assert quality["category_artifact"].endswith("v5/category.joblib")

    monkeypatch.setenv("MODEL_PROFILE", "lite")
    lite = load_runtime(tmp_path)
    assert lite["model_profile_active"] == "lite"
    assert lite["category"]["name"] == "lite"


def test_corrupted_lite_artifact_never_falls_through_to_qwen(tmp_path, monkeypatch) -> None:
    _write_stub_models(tmp_path, corrupted_lite=True)
    monkeypatch.setenv("MODEL_PROFILE", "lite")

    loaded = load_runtime(tmp_path)

    assert loaded["category"]["name"] == "legacy"
    assert loaded["model_profile_active"] == "legacy"
    assert loaded["category"]["name"] != "qwen4b"
    assert "lite V5 artifact unavailable" in loaded["model_profile_fallback_reason"]


def test_invalid_profile_is_rejected(tmp_path) -> None:
    _write_stub_models(tmp_path)

    with pytest.raises(ValueError, match="MODEL_PROFILE must be either 'quality' or 'lite'"):
        load_runtime(tmp_path, model_profile="turbo")


def test_prototype_temperature_from_artifact_is_applied(monkeypatch) -> None:
    captured: dict[str, float] = {}

    def fake_softmax(values, temperature):
        captured["temperature"] = float(temperature)
        return np.full((len(values), 2), 0.5, dtype=float)

    monkeypatch.setattr(v5_runtime, "_softmax", fake_softmax)
    pipeline = _pipeline(prototype_temperature=0.08)

    pipeline.predict_proba([{"description": "ошибка"}])

    assert captured["temperature"] == 0.08


def test_legacy_artifact_without_new_fields_uses_backward_compatible_defaults(monkeypatch) -> None:
    captured: dict[str, float] = {}
    recorder = _RecordingProbabilityModel()

    def fake_softmax(values, temperature):
        captured["temperature"] = float(temperature)
        return np.full((len(values), 2), 0.5, dtype=float)

    monkeypatch.setattr(v5_runtime, "_softmax", fake_softmax)
    pipeline = _pipeline(metadata_encoder=_ConstantMetadataEncoder(4.0), supervised_model=recorder)
    del pipeline.__dict__["metadata_scale"]
    del pipeline.__dict__["prototype_temperature"]

    probabilities = pipeline.predict_proba([{"description": "ошибка"}])

    assert probabilities.shape == (1, 2)
    assert captured["temperature"] == 0.08
    assert recorder.last_values is not None
    assert recorder.last_values[0, -1] == 4.0


def _stub_runtime() -> dict:
    class _ThreeWay:
        classes_ = np.asarray(["Прочее", "Отслеживание отправлений", "Личный кабинет"])

        def predict_proba(self, values):
            return np.tile(np.asarray([[0.7, 0.2, 0.1]], dtype=float), (len(values), 1))

    class _Routing:
        classes_ = np.asarray(["(1 линия)", "(2 линия)", "(3 линия)"])

        def predict_proba(self, values):
            return np.tile(np.asarray([[0.6, 0.3, 0.1]], dtype=float), (len(values), 1))

    class _Vectorizer:
        def transform(self, texts):
            return np.ones((len(texts), 2), dtype=float)

    return {
        "category": {
            "pipeline": _ThreeWay(),
            "top15": ["Прочее", "Отслеживание отправлений", "Личный кабинет"],
            "threshold": 0.5,
            "margin_threshold": 0.0,
            "input_contract": "registration_mapping",
            "metadata": {"model_id": "local/test", "profile": "lite"},
        },
        "routing": {"pipeline": _Routing(), "threshold": 0.55, "metadata": {}},
        "retrieval": {
            "vectorizer": _Vectorizer(),
            "matrix": np.asarray([[1.0, 0.0]]),
            "rows": [{
                "request_id": "H-1",
                "description": "похожее обращение",
                "category": "Прочее",
                "final_line": "(1 линия)",
                "result": "решено",
                "overdue": False,
                "actual_duration_seconds": 3600,
                "clarifications_count": 0,
            }],
            "rejection_threshold": 0.2,
        },
    }


def test_analyze_response_contract_is_unchanged(tmp_path) -> None:
    database = tmp_path / "posttech.db"
    connection = sqlite3.connect(database)
    connection.execute(
        "CREATE TABLE tickets (request_id TEXT, category TEXT, service TEXT, priority TEXT, "
        "final_line TEXT, overdue INTEGER, source_hash TEXT)"
    )
    connection.executemany(
        "INSERT INTO tickets VALUES (?, ?, ?, ?, ?, ?, ?)",
        [
            ("T-1", "Прочее", "ЛК", "Средний", "(1 линия)", 0, "dataset"),
            ("T-2", "Личный кабинет", "ЛК", "Средний", "(1 линия)", 1, "dataset"),
            ("T-3", "Прочее", "ЛК", "Высокий", "(1 линия)", 0, "dataset"),
        ],
    )
    connection.commit()
    connection.close()

    response = analyze_ticket(_stub_runtime(), {"description": "ошибка в личном кабинете"}, database)

    assert set(response) == {
        "category", "routing", "sla_risk", "retrieval", "similar", "model_provenance",
    }
    assert set(response["category"]) == {
        "label", "confidence", "accepted", "review_required", "threshold",
        "margin_threshold", "review_reason", "alternatives", "signals", "explanation",
    }
    assert len(response["category"]["alternatives"]) == 3
    assert all(
        {"label", "confidence"} <= set(item) for item in response["category"]["alternatives"]
    )
    assert set(response["retrieval"]) == {"rejected", "reason", "threshold", "best_relevance"}
    assert response["routing"]["label"] in {"(1 линия)", "(2 линия)", "(3 линия)"}
    assert 0.0 <= response["category"]["confidence"] <= 1.0
    assert response["model_provenance"]["model_id"] == "local/test"
