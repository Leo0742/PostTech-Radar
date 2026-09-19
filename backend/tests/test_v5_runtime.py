from __future__ import annotations

import joblib
import numpy as np
from app.ml.v5_runtime import QwenV5CategoryPipeline, build_v5_structured_features, specialist_text


class _MetadataEncoder:
    def transform(self, values):
        return np.zeros((len(values), 1), dtype=float)


class _ScaledMetadataEncoder:
    def transform(self, values):
        return np.full((len(values), 1), 4.0, dtype=float)


class _ProbabilityModel:
    classes_ = np.asarray(["A", "B"])

    def predict_proba(self, values):
        return np.tile(np.asarray([[0.8, 0.2]], dtype=float), (len(values), 1))


class _RecordingProbabilityModel:
    classes_ = np.asarray(["A", "B"])

    def __init__(self) -> None:
        self.last_values = None

    def predict_proba(self, values):
        self.last_values = np.asarray(values, dtype=float)
        return np.tile(np.asarray([[0.8, 0.2]], dtype=float), (len(values), 1))


class _KnnModel:
    classes_ = np.asarray(["A", "B"])

    def predict_proba(self, values):
        return np.tile(np.asarray([[0.6, 0.4]], dtype=float), (len(values), 1))


class _Vectorizer:
    def transform(self, texts):
        return np.zeros((len(texts), 1), dtype=float)


class _Specialist:
    classes_ = np.asarray(["A", "B"])

    def decision_function(self, values):
        return np.tile(np.asarray([[0.0, 1.0]], dtype=float), (len(values), 1))


def _pipeline(*, stub_embed: bool = True) -> QwenV5CategoryPipeline:
    runtime = QwenV5CategoryPipeline(
        labels=["A", "B"],
        model_id="local/test",
        model_revision="abc",
        instruction="",
        max_length=32,
        embedding_dim=2,
        metadata_encoder=_MetadataEncoder(),
        supervised_model=_ProbabilityModel(),
        prototype_centroids=np.asarray([[1.0, 0.0], [0.0, 1.0]]),
        knn_model=_KnnModel(),
        specialist_vectorizer=_Vectorizer(),
        specialist_model=_Specialist(),
        specialist_pairs=[("A", "B")],
        specialist_threshold=0.15,
    )
    if stub_embed:
        runtime._embed = lambda rows: np.tile(np.asarray([[1.0, 0.0]]), (len(rows), 1))  # type: ignore[method-assign]
    return runtime


def test_structured_features_tolerate_missing_or_bad_registration_date() -> None:
    values = build_v5_structured_features([
        {"description": "x"},
        {"registration_date": "not-a-date", "service": "svc"},
    ])

    assert values.shape == (2, 12)
    assert values[0, 9:].tolist() == ["__MISSING__", "__MISSING__", "__MISSING__"]
    assert values[1, 1] == "svc"


def test_specialist_text_excludes_routing_target() -> None:
    text = specialist_text({"description": "ошибка", "service": "svc", "routing_target": "LEAK"})

    assert "ошибка" in text
    assert "svc" in text
    assert "LEAK" not in text


def test_v5_pipeline_accepts_mapping_contract_and_applies_safe_pair_specialist() -> None:
    probabilities = _pipeline().predict_proba([{"description": "ошибка", "service": "svc"}])

    assert probabilities.shape == (1, 2)
    assert probabilities[0, 1] > probabilities[0, 0]
    assert _pipeline().predict([{"description": "ошибка"}]).tolist() == ["B"]


def test_v5_pipeline_applies_serialized_metadata_scale() -> None:
    runtime = _pipeline()
    recorder = _RecordingProbabilityModel()
    runtime.metadata_encoder = _ScaledMetadataEncoder()
    runtime.supervised_model = recorder
    runtime.specialist_pairs = set()
    runtime.metadata_scale = 0.25

    runtime.predict_proba([{"description": "ошибка"}])

    assert recorder.last_values is not None
    assert recorder.last_values.shape == (1, 3)
    assert recorder.last_values[0, -1] == 1.0


def test_v5_pipeline_joblib_roundtrip_preserves_lite_profile_parameters(tmp_path) -> None:
    runtime = _pipeline(stub_embed=False)
    runtime.metadata_scale = 0.75
    runtime.prototype_temperature = 0.08
    path = tmp_path / "category_qwen4b_lite.joblib"

    joblib.dump(runtime, path)
    loaded = joblib.load(path)

    assert loaded.metadata_scale == 0.75
    assert loaded.prototype_temperature == 0.08
    assert loaded.model_id == "local/test"
