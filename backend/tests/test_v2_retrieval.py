from __future__ import annotations

from app.ml.training import duplicate_free_neighbors, registration_text


def test_duplicate_free_retrieval_excludes_self_and_exact_description_group() -> None:
    scores = [1.0, 0.99, 0.8, 0.7]
    groups = ["same", "same", "other", "fourth"]
    assert duplicate_free_neighbors(scores, query_index=0, groups=groups, limit=3) == [2, 3]


def test_retrieval_query_uses_registration_fields_only() -> None:
    row = {
        "description": "QR код не работает",
        "service": "Получение",
        "component": "QR",
        "request_type": "Инцидент",
        "priority": "Высокий",
        "result": "LEAK_RESULT",
        "final_line": "LEAK_LINE",
        "overdue": "LEAK_SLA",
    }
    text = registration_text(row)
    assert "qr код" in text.lower()
    assert "LEAK_" not in text
