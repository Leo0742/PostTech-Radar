from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score


def risk_coverage_curve(
    y_true: Sequence[str],
    probabilities: np.ndarray,
    labels: Sequence[str],
    *,
    coverages: Sequence[float] = (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
) -> list[dict[str, Any]]:
    if len(y_true) != len(probabilities):
        raise ValueError("truth and probabilities must contain identical examples")
    if probabilities.ndim != 2 or probabilities.shape[1] != len(labels):
        raise ValueError("probability columns must match labels")
    confidence = probabilities.max(axis=1)
    predictions = np.asarray(labels)[probabilities.argmax(axis=1)]
    truth = np.asarray(y_true)
    order = np.argsort(-confidence, kind="stable")
    output = []
    for target in coverages:
        if not 0 < target <= 1:
            raise ValueError("coverage must be in (0, 1]")
        accepted_count = min(len(truth), max(1, math.ceil(len(truth) * target)))
        accepted = order[:accepted_count]
        correct = predictions[accepted] == truth[accepted]
        errors = int((~correct).sum())
        output.append(
            {
                "target_coverage": round(float(target), 4),
                "coverage": round(accepted_count / len(truth), 4),
                "accepted": accepted_count,
                "errors": errors,
                "accepted_accuracy": round(float(correct.mean()), 4),
            }
        )
    return output


def paired_bootstrap_accuracy(
    y_true: Sequence[str],
    predictions_a: Sequence[str],
    predictions_b: Sequence[str],
    *,
    seed: int = 20260916,
    samples: int = 2_000,
) -> dict[str, Any]:
    if not (len(y_true) == len(predictions_a) == len(predictions_b)):
        raise ValueError("paired predictions must contain identical examples")
    truth = np.asarray(y_true)
    first = np.asarray(predictions_a)
    second = np.asarray(predictions_b)
    accuracy_a = float(accuracy_score(truth, first))
    accuracy_b = float(accuracy_score(truth, second))
    random = np.random.default_rng(seed)
    differences = np.empty(samples)
    for sample in range(samples):
        indices = random.integers(0, len(truth), len(truth))
        differences[sample] = float((first[indices] == truth[indices]).mean() - (second[indices] == truth[indices]).mean())
    return {
        "paired_examples": len(truth),
        "accuracy_a": round(accuracy_a, 4),
        "accuracy_b": round(accuracy_b, 4),
        "mean_difference": round(accuracy_a - accuracy_b, 4),
        "bootstrap_ci_95": [round(float(value), 4) for value in np.quantile(differences, [0.025, 0.975])],
        "bootstrap_samples": samples,
        "seed": seed,
    }
