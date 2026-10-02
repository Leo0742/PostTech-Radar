from __future__ import annotations

import json

import joblib
import numpy as np
from app.ml.runtime import _category_prediction, load_runtime
from app.ml.v4.embedding_runtime import LazyEmbeddingClassifier
from app.ml.v4.runtime import apply_final_policy
from scipy import sparse


class _Encoder:
    def encode(self, texts, **kwargs):
        return np.asarray([[len(text), 1.0] for text in texts], dtype=np.float32)


class _Metadata:
    def transform(self, values):
        return sparse.csr_matrix([[1.0] for _ in values])


class _Classifier:
    classes_ = np.asarray(["A", "B"])

    def predict_proba(self, matrix):
        count = matrix.shape[0] if hasattr(matrix, "shape") else len(matrix)
        return np.tile(np.asarray([[0.25, 0.75]]), (count, 1))


class _BrokenClassifier:
    classes_ = np.asarray(["A", "B"])

    def predict_proba(self, values):
        raise RuntimeError("encoder weights unavailable")


def test_unknown_new_issue_is_a_final_automatic_decision() -> None:
    analysis = {
        "category": {
            "label": "Прочее",
            "confidence": 0.22,
            "accepted": False,
            "review_required": True,
            "alternatives": [],
        },
        "routing": {"label": "(2 линия)", "accepted": False, "review_required": True},
    }
    policy = {
        "unknown_label": "UNKNOWN_NEW_ISSUE",
        "known_threshold": 0.63,
        "champion": {"candidate_id": "category/structured_lr/v4"},
    }

    result = apply_final_policy(analysis, policy)

    assert result["category"]["label"] == "UNKNOWN_NEW_ISSUE"
    assert result["category"]["accepted"] is True
    assert result["category"]["review_required"] is False
    assert result["category"]["decision_source"] == "v4_unknown_gate"
    assert result["routing"]["accepted"] is True
    assert result["routing"]["review_required"] is False
    assert result["model_provenance"]["candidate_id"] == "category/structured_lr/v4"


def test_known_category_remains_final_without_manual_review() -> None:
    analysis = {
        "category": {
            "label": "Проблема с QR-код",
            "confidence": 0.91,
            "accepted": True,
            "review_required": False,
            "alternatives": [],
        }
    }
    policy = {
        "unknown_label": "UNKNOWN_NEW_ISSUE",
        "known_threshold": 0.63,
        "champion": {"candidate_id": "category/structured_lr/v4"},
    }

    result = apply_final_policy(analysis, policy)

    assert result["category"]["label"] == "Проблема с QR-код"
    assert result["category"]["accepted"] is True
    assert result["category"]["review_required"] is False
    assert result["category"]["decision_source"] == "v4_champion"


def test_runtime_loads_v4_champion_and_falls_back_if_artifact_is_broken(tmp_path) -> None:
    fallback = {"name": "v3", "top15": [], "threshold": 0.7}
    joblib.dump(fallback, tmp_path / "category.joblib")
    joblib.dump({"name": "routing"}, tmp_path / "routing.joblib")
    joblib.dump({"name": "retrieval"}, tmp_path / "retrieval.joblib")
    (tmp_path / "v4").mkdir()
    joblib.dump({"name": "v4", "top15": [], "threshold": 0.6}, tmp_path / "v4" / "category.joblib")
    (tmp_path / "v4_runtime.json").write_text(
        json.dumps({"champion": {"candidate_id": "v4"}}), encoding="utf-8"
    )

    loaded = load_runtime(tmp_path)
    assert loaded["category"]["name"] == "v4"
    assert loaded["v4_policy"]["champion"]["candidate_id"] == "v4"

    (tmp_path / "v4" / "category.joblib").write_bytes(b"not a joblib artifact")
    fallback_loaded = load_runtime(tmp_path)
    assert fallback_loaded["category"]["name"] == "v3"
    assert "v4 category unavailable" in fallback_loaded["v4_fallback_reason"]


def test_runtime_prefers_v5_bundle_over_v4_policy(tmp_path) -> None:
    fallback = {"name": "v3", "top15": [], "threshold": 0.7}
    joblib.dump(fallback, tmp_path / "category.joblib")
    joblib.dump({"name": "routing"}, tmp_path / "routing.joblib")
    joblib.dump({"name": "retrieval"}, tmp_path / "retrieval.joblib")
    (tmp_path / "v4").mkdir()
    joblib.dump({"name": "v4", "top15": [], "threshold": 0.6}, tmp_path / "v4" / "category.joblib")
    (tmp_path / "v5").mkdir()
    joblib.dump({"name": "v5", "top15": ["A"], "threshold": 0.0}, tmp_path / "v5" / "category.joblib")
    (tmp_path / "v4_runtime.json").write_text(
        json.dumps({"champion": {"candidate_id": "v4"}}), encoding="utf-8"
    )

    loaded = load_runtime(tmp_path, model_profile="quality")

    assert loaded["category"]["name"] == "v5"
    assert loaded["category_source"] == "v5"
    assert loaded["model_profile_requested"] == "quality"
    assert loaded["model_profile_active"] == "quality"
    assert "v4_policy" not in loaded


def test_runtime_lite_profile_selects_fast_lite_v2_artifact(tmp_path) -> None:
    fallback = {"name": "v3", "top15": [], "threshold": 0.7}
    joblib.dump(fallback, tmp_path / "category.joblib")
    joblib.dump({"name": "routing"}, tmp_path / "routing.joblib")
    joblib.dump({"name": "retrieval"}, tmp_path / "retrieval.joblib")
    (tmp_path / "v5").mkdir()
    joblib.dump({"name": "quality", "top15": ["A"], "threshold": 0.0}, tmp_path / "v5" / "category.joblib")
    joblib.dump(
        {"name": "lite", "top15": ["A"], "threshold": 0.0},
        tmp_path / "v5" / "category_lite_v2.joblib",
    )
    joblib.dump(
        {"name": "qwen4b", "top15": ["A"], "threshold": 0.0},
        tmp_path / "v5" / "category_qwen4b_lite.joblib",
    )

    loaded = load_runtime(tmp_path, model_profile="lite")

    assert loaded["category"]["name"] == "lite"
    assert loaded["category_source"] == "v5"
    assert loaded["model_profile_requested"] == "lite"
    assert loaded["model_profile_active"] == "lite"
    assert loaded["category_artifact"].endswith("category_lite_v2.joblib")


def test_runtime_lite_profile_does_not_silently_load_qwen_when_lite_artifact_is_missing(tmp_path) -> None:
    fallback = {"name": "v3", "top15": [], "threshold": 0.7}
    joblib.dump(fallback, tmp_path / "category.joblib")
    joblib.dump({"name": "routing"}, tmp_path / "routing.joblib")
    joblib.dump({"name": "retrieval"}, tmp_path / "retrieval.joblib")
    (tmp_path / "v5").mkdir()
    joblib.dump({"name": "quality", "top15": ["A"], "threshold": 0.0}, tmp_path / "v5" / "category.joblib")
    joblib.dump({"name": "qwen4b", "top15": ["A"], "threshold": 0.0}, tmp_path / "v5" / "category_qwen4b_lite.joblib")

    loaded = load_runtime(tmp_path, model_profile="lite")

    assert loaded["category"]["name"] == "v3"
    assert loaded["model_profile_active"] == "legacy"
    assert loaded["category"]["name"] != "qwen4b"
    assert "lite V5 artifact not found" in loaded["model_profile_fallback_reason"]


def test_embedding_runtime_uses_mapping_contract_and_cached_encoder() -> None:
    runtime = LazyEmbeddingClassifier(
        {
            "model_id": "local/test",
            "revision": "abc",
            "metadata_fields": ["service"],
            "metadata_scale": 0.5,
            "metadata_encoder": _Metadata(),
            "classifier": _Classifier(),
            "instruction": None,
        }
    )
    runtime._encoder = _Encoder()

    probabilities = runtime.predict_proba([{"description": "ошибка", "service": "mobile"}])

    assert probabilities.tolist() == [[0.25, 0.75]]
    assert runtime.predict([{"description": "ошибка", "service": "mobile"}]).tolist() == ["B"]


def test_heavy_inference_failure_uses_v3_fallback() -> None:
    fallback = {
        "pipeline": _Classifier(),
        "threshold": 0.7,
        "metadata": {"candidate_id": "v3"},
    }
    runtime = {
        "category": {
            "pipeline": _BrokenClassifier(),
            "input_contract": "registration_mapping",
        },
        "category_fallback": fallback,
    }

    bundle, alternatives, reason = _category_prediction(
        runtime, {"description": "ошибка"}, "ошибка"
    )

    assert bundle is fallback
    assert alternatives[0]["label"] == "B"
    assert "v3 fallback used" in reason
