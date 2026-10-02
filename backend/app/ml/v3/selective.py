from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np


def learn_class_thresholds(
    y_true: Sequence[str],
    probabilities: np.ndarray,
    labels: Sequence[str],
    *,
    target_accuracy: float,
    minimum_predictions: int = 5,
) -> dict[str, float]:
    predictions = np.asarray(labels)[probabilities.argmax(axis=1)]
    confidence = probabilities.max(axis=1)
    truth = np.asarray(y_true)
    thresholds: dict[str, float] = {}
    for label in labels:
        class_indices = np.where(predictions == label)[0]
        candidates = sorted({float(confidence[index]) for index in class_indices})
        valid = []
        for threshold in candidates:
            accepted = class_indices[confidence[class_indices] >= threshold]
            if len(accepted) < minimum_predictions:
                continue
            accuracy = float((truth[accepted] == predictions[accepted]).mean())
            if accuracy >= target_accuracy:
                valid.append((len(accepted), -threshold, threshold))
        thresholds[label] = round(max(valid)[2], 8) if valid else 1.0
    return thresholds


def coverage_at_accuracy_targets(
    y_true: Sequence[str],
    probabilities: np.ndarray,
    labels: Sequence[str],
    *,
    targets: Sequence[float] = (0.9, 0.95, 0.97, 0.99),
) -> dict[str, dict[str, Any]]:
    truth = np.asarray(y_true)
    predictions = np.asarray(labels)[probabilities.argmax(axis=1)]
    confidence = probabilities.max(axis=1)
    order = np.argsort(-confidence, kind="stable")
    output: dict[str, dict[str, Any]] = {}
    for target in targets:
        best_count = 0
        best_accuracy = 0.0
        for count in range(1, len(truth) + 1):
            accepted = order[:count]
            accuracy = float((truth[accepted] == predictions[accepted]).mean())
            if accuracy >= target:
                best_count = count
                best_accuracy = accuracy
        output[f"{target:.2f}"] = {
            "coverage": round(best_count / len(truth), 4),
            "accepted": best_count,
            "errors": int(round(best_count * (1 - best_accuracy))) if best_count else 0,
            "accepted_accuracy": round(best_accuracy, 4) if best_count else 0.0,
        }
    return output

