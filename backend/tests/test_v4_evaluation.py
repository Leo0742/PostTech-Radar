from __future__ import annotations

import numpy as np
from app.ml.v4.evaluation import classification_metrics, safe_registration_row


def test_full43_metrics_report_balanced_worst_and_support_bands() -> None:
    truth = ["a"] * 4 + ["b"] * 6 + ["c"] * 25 + ["d"] * 55
    predicted = ["a"] * 3 + ["b"] + ["b"] * 6 + ["c"] * 20 + ["d"] * 5 + ["d"] * 55

    metrics = classification_metrics(truth, predicted, labels=["a", "b", "c", "d"])

    assert set(metrics["support_bands"]) == {"1-5", "5-20", "20-50", ">50"}
    assert metrics["per_class"]["a"]["support"] == 4
    assert metrics["worst_class_f1"] == min(item["f1"] for item in metrics["per_class"].values())
    assert 0 <= metrics["balanced_accuracy"] <= 1


def test_registration_features_drop_every_post_resolution_field() -> None:
    row = {
        "description": "cannot login",
        "service": "portal",
        "component": "auth",
        "request_type": "incident",
        "criticality": "normal",
        "urgency": "normal",
        "priority": "medium",
        "service_class": "standard",
        "timezone": "MSK",
        "category": "secret-target",
        "final_line": "(3 линия)",
        "result": "fixed",
        "actual_duration_seconds": 5,
        "overdue": 1,
        "status": "closed",
        "clarifications_count": 3,
    }

    safe = safe_registration_row(row)

    assert safe["description"] == "cannot login"
    assert set(safe) == {
        "description",
        "service",
        "component",
        "request_type",
        "criticality",
        "urgency",
        "priority",
        "service_class",
        "timezone",
    }
    assert not np.any([value == "secret-target" for value in safe.values()])
