from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np


class APSConformalClassifier:
    def __init__(self, alpha: float = 0.05):
        if not 0 < alpha < 1:
            raise ValueError("alpha must be in (0, 1)")
        self.alpha = alpha
        self.labels: tuple[str, ...] = ()
        self.quantile: float | None = None

    @staticmethod
    def _aps_scores(y_true: Sequence[str], probabilities: np.ndarray, labels: Sequence[str]) -> np.ndarray:
        label_to_index = {label: index for index, label in enumerate(labels)}
        scores = []
        for truth, row in zip(y_true, probabilities, strict=True):
            order = np.argsort(-row, kind="stable")
            cumulative = np.cumsum(row[order])
            position = int(np.where(order == label_to_index[truth])[0][0])
            scores.append(float(cumulative[position]))
        return np.asarray(scores)

    def fit(self, y_true: Sequence[str], probabilities: np.ndarray, labels: Sequence[str]) -> APSConformalClassifier:
        if len(y_true) != len(probabilities):
            raise ValueError("calibration truth and probabilities must align")
        self.labels = tuple(labels)
        scores = self._aps_scores(y_true, probabilities, labels)
        level = min(1.0, math.ceil((len(scores) + 1) * (1 - self.alpha)) / len(scores))
        self.quantile = float(np.quantile(scores, level, method="higher"))
        return self

    def predict_sets(self, probabilities: np.ndarray) -> list[list[str]]:
        if self.quantile is None:
            raise RuntimeError("fit must be called before predict_sets")
        output = []
        for row in probabilities:
            order = np.argsort(-row, kind="stable")
            cumulative = np.cumsum(row[order])
            stop = int(np.searchsorted(cumulative, self.quantile, side="left"))
            selected = order[: min(len(order), stop + 1)]
            output.append([self.labels[index] for index in selected])
        return output

    def summary(self, y_true: Sequence[str], probabilities: np.ndarray) -> dict[str, Any]:
        prediction_sets = self.predict_sets(probabilities)
        covered = [truth in values for truth, values in zip(y_true, prediction_sets, strict=True)]
        singleton = [len(values) == 1 for values in prediction_sets]
        singleton_correct = [truth == values[0] for truth, values in zip(y_true, prediction_sets, strict=True) if len(values) == 1]
        return {
            "method": "APS",
            "alpha": self.alpha,
            "target_coverage": round(1 - self.alpha, 4),
            "empirical_coverage": round(float(np.mean(covered)), 4),
            "average_set_size": round(float(np.mean([len(values) for values in prediction_sets])), 4),
            "singleton_rate": round(float(np.mean(singleton)), 4),
            "singleton_accuracy": round(float(np.mean(singleton_correct)), 4) if singleton_correct else 0.0,
            "quantile": round(float(self.quantile), 8),
        }

