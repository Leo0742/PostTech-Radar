from __future__ import annotations

import joblib
import numpy as np
from app.ml.lite_v2_runtime import LiteV2CategoryPipeline
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import LinearSVC


def _rows():
    return [
        {"description": "не могу войти в личный кабинет", "service": "ЛК", "priority": "Средний"},
        {"description": "не приходит push уведомление", "service": "Уведомления", "priority": "Средний"},
        {"description": "трек номер не обновляется", "service": "Отслеживание", "priority": "Высокий"},
        {"description": "личный кабинет не открывается", "service": "ЛК", "priority": "Высокий"},
    ]


def _pipeline() -> LiteV2CategoryPipeline:
    rows = _rows()
    labels = ["Личный кабинет", "push/sms/email", "Отслеживание"]
    y = ["Личный кабинет", "push/sms/email", "Отслеживание", "Личный кабинет"]
    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=1)
    text = vectorizer.fit_transform([row["description"] for row in rows])
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    metadata = encoder.fit_transform([[row.get("service", ""), row.get("priority", "")] for row in rows])
    x = sparse.hstack([text, metadata * 0.35], format="csr")
    model = LinearSVC(C=0.35, class_weight="balanced", random_state=1).fit(x, y)
    return LiteV2CategoryPipeline(
        labels=labels,
        vectorizer=vectorizer,
        metadata_encoder=encoder,
        classifier=model,
        metadata_fields=("service", "priority"),
        metadata_scale=0.35,
        temperature=1.0,
        text_mode="description",
    )


def test_lite_v2_pipeline_accepts_registration_mapping_and_normalizes_probabilities() -> None:
    pipeline = _pipeline()

    probabilities = pipeline.predict_proba([{"description": "не могу войти", "service": "ЛК"}])

    assert probabilities.shape == (1, 3)
    assert np.isclose(probabilities.sum(axis=1), 1.0).all()
    assert list(pipeline.classes_) == ["Личный кабинет", "push/sms/email", "Отслеживание"]


def test_lite_v2_pipeline_survives_joblib_roundtrip_and_partial_input(tmp_path) -> None:
    path = tmp_path / "lite.joblib"
    joblib.dump(_pipeline(), path)
    restored = joblib.load(path)

    probabilities = restored.predict_proba([{"description": "трек не обновляется"}])

    assert probabilities.shape == (1, 3)
    assert np.isfinite(probabilities).all()
    assert np.isclose(probabilities.sum(axis=1), 1.0).all()


class _SpecialistVectorizer:
    def transform(self, texts):
        return np.zeros((len(texts), 1), dtype=float)


class _SpecialistModel:
    def __init__(self, labels, preferred):
        self.classes_ = np.asarray(labels, dtype=object)
        self.preferred = str(preferred)

    def decision_function(self, values):
        row = np.zeros(len(self.classes_), dtype=float)
        row[list(self.classes_).index(self.preferred)] = 1.0
        return np.tile(row.reshape(1, -1), (len(values), 1))


def test_lite_v2_specialist_can_swap_only_a_frozen_top2_pair() -> None:
    pipeline = _pipeline()
    row = {"description": "не могу войти", "service": "ЛК"}
    before = pipeline.predict_proba([row])
    order = np.argsort(-before[0])
    first, second = pipeline.labels[int(order[0])], pipeline.labels[int(order[1])]

    pipeline.specialist_vectorizer = _SpecialistVectorizer()
    pipeline.specialist_model = _SpecialistModel(pipeline.labels, second)
    pipeline.specialist_pairs = {frozenset((first, second))}
    pipeline.specialist_threshold = 0.1

    after = pipeline.predict_proba([row])

    assert pipeline.labels[int(np.argmax(before[0]))] == first
    assert pipeline.labels[int(np.argmax(after[0]))] == second
