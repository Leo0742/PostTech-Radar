from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from scipy import sparse

from app.ml.v5_runtime import build_v5_structured_features, specialist_text


def _softmax(values: np.ndarray, temperature: float) -> np.ndarray:
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim == 1:
        matrix = np.column_stack([-matrix, matrix])
    scaled = matrix / max(float(temperature), 1e-6)
    scaled -= np.max(scaled, axis=1, keepdims=True)
    exp = np.exp(scaled)
    return exp / exp.sum(axis=1, keepdims=True)


class LiteV2CategoryPipeline:
    """Small CPU category model used by the default local runtime.

    The model combines sparse TF-IDF text features with the same structured
    registration metadata branch used during V5 development.  LinearSVC
    scores are converted to normalized probabilities with one frozen
    temperature selected only from development OOF predictions.
    """

    def __init__(
        self,
        *,
        labels: Sequence[str],
        vectorizer: Any,
        metadata_encoder: Any,
        classifier: Any,
        metadata_scale: float,
        temperature: float,
        text_mode: str = "specialist",
        metadata_fields: Sequence[str] | None = None,
        specialist_vectorizer: Any | None = None,
        specialist_model: Any | None = None,
        specialist_pairs: Sequence[Sequence[str]] = (),
        specialist_threshold: float = 0.1,
    ) -> None:
        if text_mode not in {"description", "specialist"}:
            raise ValueError("text_mode must be 'description' or 'specialist'")
        self.labels = [str(value) for value in labels]
        self.vectorizer = vectorizer
        self.metadata_encoder = metadata_encoder
        self.classifier = classifier
        self.metadata_scale = float(metadata_scale)
        self.temperature = float(temperature)
        self.text_mode = text_mode
        self.metadata_fields = tuple(metadata_fields) if metadata_fields is not None else None
        self.specialist_vectorizer = specialist_vectorizer
        self.specialist_model = specialist_model
        self.specialist_pairs = {frozenset(map(str, pair)) for pair in specialist_pairs}
        self.specialist_threshold = float(specialist_threshold)

    @property
    def classes_(self) -> np.ndarray:
        return np.asarray(self.labels, dtype=object)

    def _texts(self, rows: Sequence[Mapping[str, Any]]) -> list[str]:
        if self.text_mode == "description":
            return [" ".join(str(row.get("description") or "").split()) for row in rows]
        return [specialist_text(row) for row in rows]

    def _metadata(self, rows: Sequence[Mapping[str, Any]]) -> Any:
        if self.metadata_fields is None:
            values = build_v5_structured_features(rows)
        else:
            values = np.asarray(
                [[str(row.get(field) or "") for field in self.metadata_fields] for row in rows],
                dtype=object,
            )
        return self.metadata_encoder.transform(values)

    def _apply_specialist(
        self,
        rows: Sequence[Mapping[str, Any]],
        probabilities: np.ndarray,
    ) -> np.ndarray:
        vectorizer = getattr(self, "specialist_vectorizer", None)
        model = getattr(self, "specialist_model", None)
        pairs = getattr(self, "specialist_pairs", set())
        if vectorizer is None or model is None or not pairs:
            return probabilities

        matrix = vectorizer.transform([specialist_text(row) for row in rows])
        decision = np.asarray(model.decision_function(matrix), dtype=float)
        if decision.ndim == 1:
            decision = np.column_stack([-decision, decision])
        classes = [str(value) for value in model.classes_]
        class_index = {label: index for index, label in enumerate(classes)}
        threshold = float(getattr(self, "specialist_threshold", 0.1))
        result = np.asarray(probabilities, dtype=float).copy()

        for row_index in range(len(rows)):
            order = np.argsort(-result[row_index])
            if len(order) < 2:
                continue
            first = self.labels[int(order[0])]
            second = self.labels[int(order[1])]
            if frozenset((first, second)) not in pairs:
                continue
            if first not in class_index or second not in class_index:
                continue
            margin = decision[row_index, class_index[second]] - decision[row_index, class_index[first]]
            if margin > threshold:
                left, right = int(order[0]), int(order[1])
                result[row_index, left], result[row_index, right] = result[row_index, right], result[row_index, left]
        return result

    def predict_proba(self, rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
        text_values = self.vectorizer.transform(self._texts(rows))
        metadata_values = self._metadata(rows)
        values = sparse.hstack([text_values, metadata_values * self.metadata_scale], format="csr")
        local_probabilities = _softmax(self.classifier.decision_function(values), self.temperature)

        result = np.zeros((len(rows), len(self.labels)), dtype=float)
        positions = {label: index for index, label in enumerate(self.labels)}
        for source, label in enumerate(self.classifier.classes_):
            target = positions.get(str(label))
            if target is not None:
                result[:, target] = local_probabilities[:, source]
        sums = result.sum(axis=1, keepdims=True)
        normalized = np.divide(result, sums, out=np.zeros_like(result), where=sums > 0)
        return self._apply_specialist(rows, normalized)
