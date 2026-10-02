from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


def monitoring_event(
    *,
    request_id: str,
    model_version: str,
    scores: Mapping[str, float],
    accepted: bool,
    oos_score: float,
    prediction_set: Sequence[str],
    corrected_label: str | None = None,
) -> dict[str, Any]:
    ordered_scores = sorted(((str(label), float(score)) for label, score in scores.items()), key=lambda item: -item[1])
    return {
        "request_id": str(request_id),
        "model_version": model_version,
        "top_scores": [{"label": label, "score": round(score, 8)} for label, score in ordered_scores[:5]],
        "accepted": bool(accepted),
        "review_required": not accepted,
        "oos_score": round(float(oos_score), 8),
        "prediction_set": list(prediction_set),
        "operator_corrected_label": corrected_label,
        "human_confirmed": corrected_label is not None,
    }


def _distribution(labels: Sequence[str], vocabulary: Sequence[str]) -> np.ndarray:
    counts = Counter(str(label) for label in labels)
    values = np.asarray([counts[label] for label in vocabulary], dtype=float) + 1e-9
    return values / values.sum()


def drift_summary(
    *,
    baseline_labels: Sequence[str],
    current_labels: Sequence[str],
    baseline_accepted: Sequence[bool],
    current_accepted: Sequence[bool],
    js_alert_threshold: float = 0.1,
    abstention_alert_threshold: float = 0.15,
) -> dict[str, Any]:
    if not baseline_labels or not current_labels:
        raise ValueError("baseline and current labels are required")
    vocabulary = sorted(set(map(str, baseline_labels)) | set(map(str, current_labels)))
    baseline = _distribution(baseline_labels, vocabulary)
    current = _distribution(current_labels, vocabulary)
    midpoint = 0.5 * (baseline + current)
    js = 0.5 * np.sum(baseline * np.log2(baseline / midpoint)) + 0.5 * np.sum(current * np.log2(current / midpoint))
    baseline_abstention = 1.0 - float(np.mean(baseline_accepted))
    current_abstention = 1.0 - float(np.mean(current_accepted))
    change = current_abstention - baseline_abstention
    return {
        "labels": vocabulary,
        "jensen_shannon_divergence": round(float(js), 6),
        "baseline_abstention_rate": round(baseline_abstention, 6),
        "current_abstention_rate": round(current_abstention, 6),
        "abstention_rate_change": round(change, 6),
        "alert": bool(js >= js_alert_threshold or abs(change) >= abstention_alert_threshold),
        "recommended_action": "review rejected clusters and confirmed operator labels before retraining",
    }

