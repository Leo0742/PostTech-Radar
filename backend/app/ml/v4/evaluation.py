from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, f1_score

REGISTRATION_FIELDS = (
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


def safe_registration_row(row: dict[str, Any]) -> dict[str, str]:
    return {field: str(row.get(field) or "") for field in REGISTRATION_FIELDS}


def _support_band(support: int) -> str:
    if support <= 5:
        return "1-5"
    if support <= 20:
        return "5-20"
    if support <= 50:
        return "20-50"
    return ">50"


def classification_metrics(
    truth: Sequence[str],
    predicted: Sequence[str],
    *,
    labels: Sequence[str],
) -> dict[str, Any]:
    if len(truth) != len(predicted):
        raise ValueError("truth and predicted must have equal length")
    label_list = list(labels)
    report = classification_report(
        truth,
        predicted,
        labels=label_list,
        output_dict=True,
        zero_division=0,
    )
    supports = Counter(truth)
    per_class = {
        label: {
            "precision": round(float(report[label]["precision"]), 6),
            "recall": round(float(report[label]["recall"]), 6),
            "f1": round(float(report[label]["f1-score"]), 6),
            "support": int(supports[label]),
        }
        for label in label_list
    }
    bands: dict[str, list[float]] = {"1-5": [], "5-20": [], "20-50": [], ">50": []}
    for _label, values in per_class.items():
        support = int(values["support"])
        if support:
            bands[_support_band(support)].append(float(values["f1"]))
    support_bands = {
        name: {
            "macro_f1": round(float(np.mean(values)), 6) if values else None,
            "class_count": len(values),
        }
        for name, values in bands.items()
    }
    class_f1 = [float(item["f1"]) for item in per_class.values()]
    return {
        "rows": len(truth),
        "accuracy": round(float(accuracy_score(truth, predicted)), 6),
        "macro_f1": round(float(f1_score(truth, predicted, labels=label_list, average="macro", zero_division=0)), 6),
        "weighted_f1": round(
            float(f1_score(truth, predicted, labels=label_list, average="weighted", zero_division=0)), 6
        ),
        "balanced_accuracy": round(float(balanced_accuracy_score(truth, predicted)), 6),
        "worst_class_f1": round(min(class_f1), 6) if class_f1 else 0.0,
        "per_class": per_class,
        "support_bands": support_bands,
    }
