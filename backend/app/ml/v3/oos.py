from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def probability_oos_scores(probabilities: np.ndarray) -> list[dict[str, float]]:
    clipped = np.clip(probabilities, 1e-12, 1.0)
    sorted_probabilities = np.sort(clipped, axis=1)
    entropy = -np.sum(clipped * np.log(clipped), axis=1) / np.log(probabilities.shape[1])
    return [
        {
            "maximum_probability": round(float(row[-1]), 8),
            "margin": round(float(row[-1] - row[-2]), 8) if len(row) > 1 else round(float(row[-1]), 8),
            "normalized_entropy": round(float(entropy[index]), 8),
        }
        for index, row in enumerate(sorted_probabilities)
    ]


def threshold_for_oos_recall(
    scores: Sequence[float], is_oos: Sequence[bool], *, target_recall: float
) -> float:
    if len(scores) != len(is_oos):
        raise ValueError("scores and labels must align")
    values = np.asarray(scores, dtype=float)
    truth = np.asarray(is_oos, dtype=bool)
    if not truth.any():
        raise ValueError("at least one OOS example is required")
    valid = []
    for threshold in sorted(set(values.tolist())):
        predicted = values >= threshold
        recall = float((predicted & truth).sum() / truth.sum())
        false_accepts = int((predicted & ~truth).sum())
        if recall >= target_recall:
            valid.append((false_accepts, -threshold, threshold))
    if not valid:
        raise ValueError("target recall is not attainable")
    return round(float(min(valid)[2]), 8)

