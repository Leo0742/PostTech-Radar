from __future__ import annotations

import joblib
import numpy as np
from app.ml.lite_v3_runtime import LiteV3DistilledCategoryPipeline
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import LinearSVC


class _IdentityProjector:
    def predict(self, values):
        return np.asarray(values, dtype=float)


class _FakeSentenceModel:
    def encode(self, texts, **_kwargs):
        result = []
        for text in texts:
            lowered = text.lower()
            if "кабинет" in lowered:
                result.append([1.0, 0.0])
            elif "push" in lowered:
                result.append([0.0, 1.0])
            else:
                result.append([0.5, 0.5])
        return np.asarray(result, dtype=np.float32)


def _pipeline() -> LiteV3DistilledCategoryPipeline:
    rows = [
        {"description": "не могу войти в личный кабинет", "service": "ЛК"},
        {"description": "личный кабинет не открывается", "service": "ЛК"},
        {"description": "не приходит push", "service": "Уведомления"},
        {"description": "push уведомление пропало", "service": "Уведомления"},
    ]
    y = ["Личный кабинет", "Личный кабинет", "push/sms/email", "push/sms/email"]
    labels = ["Личный кабинет", "push/sms/email"]
    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=1)
    text = vectorizer.fit_transform([row["description"] for row in rows])
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    metadata = encoder.fit_transform([[row["service"]] for row in rows])
    semantic = sparse.csr_matrix(np.asarray([[1, 0], [1, 0], [0, 1], [0, 1]], dtype=float))
    matrix = sparse.hstack([text, metadata * 0.35, semantic], format="csr")
    classifier = LinearSVC(C=0.35, class_weight="balanced", random_state=1).fit(matrix, y)
    pipeline = LiteV3DistilledCategoryPipeline(
        labels=labels,
        vectorizer=vectorizer,
        metadata_encoder=encoder,
        classifier=classifier,
        metadata_fields=("service",),
        metadata_scale=0.35,
        temperature=1.0,
        text_mode="description",
        projector=_IdentityProjector(),
        minilm_model_id="fake/minilm",
        local_model_dir="unused",
    )
    pipeline._sentence_model = _FakeSentenceModel()
    return pipeline


def test_lite_v3_distilled_pipeline_returns_normalized_probabilities() -> None:
    pipeline = _pipeline()
    probabilities = pipeline.predict_proba([{"description": "личный кабинет", "service": "ЛК"}])

    assert probabilities.shape == (1, 2)
    assert np.isclose(probabilities.sum(axis=1), 1.0).all()
    assert pipeline.classes_[int(np.argmax(probabilities[0]))] == "Личный кабинет"


def test_lite_v3_does_not_serialize_loaded_sentence_model(tmp_path) -> None:
    pipeline = _pipeline()
    path = tmp_path / "lite_v3.joblib"
    joblib.dump(pipeline, path)
    restored = joblib.load(path)

    assert restored._sentence_model is None
