from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

SAFE_FEATURES: tuple[str, ...] = (
    "description",
    "service",
    "component",
    "request_type",
    "criticality",
    "urgency",
    "priority",
    "service_class",
    "timezone",
)


@dataclass(frozen=True)
class CandidateSpec:
    candidate_id: str
    task: str
    family: str
    feature_mode: str = "text"
    tier: str = "A"
    parameters: Mapping[str, Any] = field(default_factory=dict)


def safe_feature_row(row: Mapping[str, Any]) -> dict[str, str]:
    return {name: str(row.get(name) or "") for name in SAFE_FEATURES}


class TextSelector(BaseEstimator, TransformerMixin):
    def fit(self, values: Sequence[Mapping[str, str]], y: Any = None) -> TextSelector:
        return self

    def transform(self, values: Sequence[Mapping[str, str]]) -> list[str]:
        return [str(value.get("description") or "") for value in values]

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        return np.asarray(["description"], dtype=object)


class MetadataSelector(BaseEstimator, TransformerMixin):
    def fit(self, values: Sequence[Mapping[str, str]], y: Any = None) -> MetadataSelector:
        return self

    def transform(self, values: Sequence[Mapping[str, str]]) -> list[dict[str, str]]:
        return [{name: str(value.get(name) or "") for name in SAFE_FEATURES if name != "description"} for value in values]

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        return np.asarray([name for name in SAFE_FEATURES if name != "description"], dtype=object)


class CatBoostTextClassifier:
    def __init__(self, parameters: Mapping[str, Any]):
        self.parameters = dict(parameters)
        self.model: Any = None
        self.classes_: np.ndarray = np.asarray([])

    @staticmethod
    def _frame(values: Sequence[Mapping[str, str]]) -> Any:
        import pandas as pd

        return pd.DataFrame([{name: str(value.get(name) or "") for name in SAFE_FEATURES} for value in values])

    def fit(self, values: Sequence[Mapping[str, str]], labels: Sequence[str]) -> CatBoostTextClassifier:
        from catboost import CatBoostClassifier

        iterations = int(self.parameters.get("iterations", 250))
        self.model = CatBoostClassifier(
            iterations=iterations,
            depth=int(self.parameters.get("depth", 6)),
            learning_rate=float(self.parameters.get("learning_rate", 0.05)),
            loss_function="MultiClass",
            auto_class_weights="Balanced",
            random_seed=int(self.parameters.get("seed", 20260916)),
            verbose=False,
            allow_writing_files=False,
            thread_count=int(self.parameters.get("thread_count", 4)),
        )
        categorical = [name for name in SAFE_FEATURES if name != "description"]
        self.model.fit(self._frame(values), list(labels), cat_features=categorical, text_features=["description"])
        self.classes_ = np.asarray(self.model.classes_)
        return self

    def predict(self, values: Sequence[Mapping[str, str]]) -> np.ndarray:
        return np.asarray(self.model.predict(self._frame(values))).reshape(-1)

    def predict_proba(self, values: Sequence[Mapping[str, str]]) -> np.ndarray:
        return np.asarray(self.model.predict_proba(self._frame(values)))


def _word_char_features() -> FeatureUnion:
    return FeatureUnion(
        [
            (
                "word",
                TfidfVectorizer(
                    lowercase=True,
                    ngram_range=(1, 3),
                    min_df=1,
                    max_df=0.995,
                    max_features=24_000,
                    sublinear_tf=True,
                ),
            ),
            (
                "char",
                TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=(2, 6),
                    min_df=1,
                    max_features=32_000,
                    sublinear_tf=True,
                ),
            ),
        ]
    )


def _feature_pipeline(mode: str) -> Any:
    text = Pipeline([("select", TextSelector()), ("vectorize", _word_char_features())])
    metadata = Pipeline([("select", MetadataSelector()), ("vectorize", DictVectorizer(sparse=True, sort=True))])
    if mode == "text":
        return text
    if mode == "metadata":
        return metadata
    if mode == "combined":
        return FeatureUnion([("text", text), ("metadata", metadata)])
    raise ValueError(f"Unknown feature mode: {mode}")


def build_estimator(spec: CandidateSpec) -> Any:
    if spec.family == "catboost_text":
        return CatBoostTextClassifier(spec.parameters)
    features = _feature_pipeline(spec.feature_mode)
    if spec.family in {"tfidf_lr", "structured_lr"}:
        classifier: Any = LogisticRegression(
            max_iter=int(spec.parameters.get("max_iter", 1_500)),
            C=float(spec.parameters.get("C", 2.0)),
            class_weight=spec.parameters.get("class_weight", "balanced"),
            random_state=int(spec.parameters.get("seed", 20260916)),
        )
    elif spec.family in {"linear_svc", "calibrated_linear_svc"}:
        classifier = LinearSVC(
            C=float(spec.parameters.get("C", 0.8)),
            class_weight="balanced",
            random_state=int(spec.parameters.get("seed", 20260916)),
        )
    else:
        raise ValueError(f"Unknown candidate family: {spec.family}")
    return Pipeline([("features", features), ("classifier", classifier)])


class GroupCalibratedLinearSVC:
    def __init__(self, estimator: Pipeline, calibrator: LogisticRegression):
        self.estimator = estimator
        self.calibrator = calibrator
        self.classes_ = calibrator.classes_

    @staticmethod
    def _scores(estimator: Pipeline, values: Sequence[Mapping[str, str]]) -> np.ndarray:
        scores = np.asarray(estimator.decision_function(values))
        return scores.reshape(-1, 1) if scores.ndim == 1 else scores

    def predict_proba(self, values: Sequence[Mapping[str, str]]) -> np.ndarray:
        return self.calibrator.predict_proba(self._scores(self.estimator, values))

    def predict(self, values: Sequence[Mapping[str, str]]) -> np.ndarray:
        return self.calibrator.predict(self._scores(self.estimator, values))


def fit_group_calibrated_svc(
    spec: CandidateSpec,
    values: Sequence[Mapping[str, str]],
    labels: Sequence[str],
    groups: Sequence[str],
    *,
    seed: int,
) -> tuple[GroupCalibratedLinearSVC, dict[str, Any]]:
    if not (len(values) == len(labels) == len(groups)):
        raise ValueError("values, labels and groups must have equal length")
    groups_per_class: dict[str, set[str]] = {}
    for label, group in zip(labels, groups, strict=True):
        groups_per_class.setdefault(str(label), set()).add(str(group))
    minimum = min(len(items) for items in groups_per_class.values())
    if minimum < 2:
        raise ValueError("Calibration needs at least two independent groups per class")
    splits = min(4, minimum)
    indices = np.arange(len(values))
    splitter = StratifiedGroupKFold(n_splits=splits, shuffle=True, random_state=seed)
    fit_indices, calibration_indices = next(splitter.split(indices, labels, groups))
    base = build_estimator(spec)
    base.fit([values[index] for index in fit_indices], [labels[index] for index in fit_indices])
    calibration_values = [values[index] for index in calibration_indices]
    scores = GroupCalibratedLinearSVC._scores(base, calibration_values)
    calibrator = LogisticRegression(max_iter=1_000, random_state=seed)
    calibrator.fit(
        scores,
        [labels[index] for index in calibration_indices],
    )
    calibrated = GroupCalibratedLinearSVC(base, calibrator)
    return calibrated, {
        "method": "multinomial_logistic_on_group_disjoint_decision_scores",
        "seed": seed,
        "fit_count": len(fit_indices),
        "calibration_count": len(calibration_indices),
        "fit_groups": sorted({str(groups[index]) for index in fit_indices}),
        "calibration_groups": sorted({str(groups[index]) for index in calibration_indices}),
    }
