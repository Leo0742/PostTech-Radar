from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
from app.ml.runtime import analyze_ticket, retrieve_similar_tickets
from app.ml.training import registration_text
from app.services.data_service import ensure_database_schema
from sklearn.feature_extraction.text import TfidfVectorizer


class _FixedProbabilityModel:
    def __init__(self, classes: list[str], probabilities: list[float]):
        self.classes_ = np.asarray(classes)
        self._probabilities = np.asarray(probabilities)

    def predict_proba(self, values: list[str]) -> np.ndarray:
        return np.tile(self._probabilities, (len(values), 1))


def _seed_sla_database(path: Path) -> None:
    ensure_database_schema(path)
    connection = sqlite3.connect(path)
    for index in range(40):
        values = (
            f"T-{index}", "2026-01-01T00:00:00", None, "S", None, "A", "incident", f"known {index}",
            f"known {index}", "medium", "medium", "medium", "standard", "MSK", "closed", 0,
            "L1", 60, 0, 0, 1, 0, 0, 0, 1, "done", "{}", "test", "2026-01-01T00:00:00",
        )
        connection.execute(
            "INSERT INTO tickets VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            values,
        )
    connection.commit()
    connection.close()


def _runtime() -> dict[str, object]:
    rows = [
        {
            "request_id": "R-1",
            "description": "не работает qr код",
            "category": "A",
            "final_line": "L1",
            "result": "done",
            "overdue": 0,
            "actual_duration_seconds": 60,
            "clarifications_count": 0,
            "service": "S",
            "component": "QR",
            "request_type": "incident",
            "priority": "medium",
        },
        {
            "request_id": "R-2",
            "description": "ошибка архива zip",
            "category": "B",
            "final_line": "L2",
            "result": "done",
            "overdue": 0,
            "actual_duration_seconds": 120,
            "clarifications_count": 0,
            "service": "S",
            "component": "ZIP",
            "request_type": "incident",
            "priority": "medium",
        },
    ]
    retrieval_texts = [registration_text(row) for row in rows]
    vectorizer = TfidfVectorizer().fit(retrieval_texts)
    return {
        "category": {
            "pipeline": _FixedProbabilityModel(["A", "B"], [0.51, 0.49]),
            "explanation_pipeline": None,
            "threshold": 0.9,
            "margin_threshold": 0.2,
        },
        "routing": {
            "pipeline": _FixedProbabilityModel(["L1", "L2"], [0.52, 0.48]),
            "explanation_pipeline": None,
            "threshold": 0.9,
        },
        "retrieval": {
            "vectorizer": vectorizer,
            "matrix": vectorizer.transform(retrieval_texts),
            "rows": rows,
            "metadata_boost": False,
            "rejection_threshold": 0.2,
        },
    }


def test_rejected_predictions_do_not_segment_sla_and_weak_retrieval_is_empty(tmp_path: Path) -> None:
    database = tmp_path / "runtime.db"
    _seed_sla_database(database)
    payload = {
        "description": "абракадабра квантовый телепорт океан",
        "service": "",
        "component": "",
        "request_type": "",
        "criticality": "",
        "urgency": "",
        "priority": "",
        "service_class": "",
        "timezone": "",
    }

    result = analyze_ticket(_runtime(), payload, database)

    assert result["category"]["accepted"] is False
    assert result["routing"]["accepted"] is False
    assert result["routing"]["explanation"] == (
        "Линия выбрана по регистрационным признакам обращения."
    )
    assert "4-я линия" not in result["routing"]["explanation"]
    assert result["sla_risk"]["basis"] == "общий уровень"
    assert result["similar"] == []
    assert result["retrieval"] == {
        "rejected": True,
        "reason": "Достаточно похожих исторических обращений не найдено.",
        "threshold": 0.2,
        "best_relevance": 0.0,
    }


def test_retrieval_returns_items_when_best_relevance_reaches_threshold() -> None:
    payload = {
        "description": "не работает qr код",
        "service": "S",
        "component": "QR",
        "request_type": "incident",
        "criticality": "",
        "urgency": "",
        "priority": "medium",
        "service_class": "",
        "timezone": "",
    }

    result = retrieve_similar_tickets(_runtime(), payload, limit=1)

    assert result["rejected"] is False
    assert result["reason"] == ""
    assert result["best_relevance"] == 1.0
    assert [item["request_id"] for item in result["items"]] == ["R-1"]
