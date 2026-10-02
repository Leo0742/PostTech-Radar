from __future__ import annotations

from app.ml.v3.monitoring import drift_summary, monitoring_event


def test_monitoring_event_keeps_learning_signals_without_ticket_text() -> None:
    event = monitoring_event(
        request_id="42",
        model_version="v3",
        scores={"A": 0.7, "B": 0.3},
        accepted=False,
        oos_score=0.8,
        prediction_set=["A", "B"],
        corrected_label="B",
    )
    assert event["request_id"] == "42"
    assert "description" not in event
    assert event["human_confirmed"] is True


def test_drift_summary_detects_frequency_and_abstention_shift() -> None:
    result = drift_summary(
        baseline_labels=["A"] * 8 + ["B"] * 2,
        current_labels=["A"] * 2 + ["B"] * 8,
        baseline_accepted=[True] * 9 + [False],
        current_accepted=[True] * 4 + [False] * 6,
    )
    assert result["jensen_shannon_divergence"] > 0.1
    assert result["abstention_rate_change"] == 0.5
    assert result["alert"] is True

