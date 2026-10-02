from __future__ import annotations

import json
import sqlite3
from io import BytesIO

import numpy as np
from app.api import routes as routes_module
from app.main import app
from app.ml.runtime import route_ticket
from app.services.incoming_service import (
    complete_ticket,
    confirm_category,
    confirm_route,
    create_batch_from_excel,
    list_batch_tickets,
    save_analysis,
    save_routing_recommendation,
    ticket_analysis_payload,
)
from fastapi.testclient import TestClient
from openpyxl import Workbook


class FixedRoutingModel:
    classes_ = np.asarray(["(1 линия)", "(2 линия)", "(3 линия)"])

    def predict_proba(self, values):  # noqa: ANN001
        return np.asarray([[0.45, 0.40, 0.15] for _ in values], dtype=float)



def _workbook(headers: list[str], rows: list[list[object]]) -> bytes:
    workbook = Workbook()
    sheet = workbook.active
    sheet.append(headers)
    for row in rows:
        sheet.append(row)
    stream = BytesIO()
    workbook.save(stream)
    return stream.getvalue()



def _analysis() -> dict[str, object]:
    return {
        "category": {
            "label": "Категория A",
            "confidence": 0.51,
            "review_required": True,
            "alternatives": [
                {"label": "Категория A", "confidence": 0.51},
                {"label": "Категория B", "confidence": 0.46},
                {"label": "Категория C", "confidence": 0.03},
            ],
        },
        "routing": {
            "label": "(1 линия)",
            "confidence": 0.45,
            "review_required": True,
            "alternatives": [
                {"label": "(1 линия)", "confidence": 0.45},
                {"label": "(2 линия)", "confidence": 0.40},
                {"label": "(3 линия)", "confidence": 0.15},
            ],
        },
        "sla_risk": {"risk": 0.1, "sample_size": 8, "level": "Низкий", "explanation": ""},
        "similar": [],
        "model_provenance": {"version": "v5.2", "model_id": "Qwen/Qwen3-Embedding-8B"},
    }


def _seed_historical_routes(database) -> None:  # noqa: ANN001
    connection = sqlite3.connect(database)
    connection.execute(
        "CREATE TABLE tickets (request_id TEXT PRIMARY KEY, category TEXT, final_line TEXT, source_hash TEXT)"
    )
    rows = []
    for index in range(20):
        rows.append((f"A-{index}", "Категория A", "(2 линия)" if index < 18 else "(1 линия)", "historical"))
    for index in range(20):
        rows.append((f"B-{index}", "Категория B", "(1 линия)" if index < 18 else "(2 линия)", "historical"))
    connection.executemany("INSERT INTO tickets VALUES (?, ?, ?, ?)", rows)
    connection.commit()
    connection.close()


def test_routing_uses_topk_before_confirmation_and_confirmed_category_after(tmp_path) -> None:
    database = tmp_path / "routing.sqlite3"
    _seed_historical_routes(database)
    runtime = {"routing": {"pipeline": FixedRoutingModel(), "threshold": 0.55}}
    payload = {"description": "пример"}

    before = route_ticket(
        runtime,
        payload,
        category_alternatives=[
            {"label": "Категория A", "confidence": 0.8},
            {"label": "Категория B", "confidence": 0.2},
        ],
        database=database,
    )
    after = route_ticket(runtime, payload, confirmed_category="Категория B", database=database)

    assert before["mode"] == "category_topk"
    assert before["label"] == "(2 линия)"
    assert after["mode"] == "confirmed_category"
    assert after["label"] == "(1 линия)"
    assert len(before["alternatives"]) == 3
    assert len(after["alternatives"]) == 3


def test_completed_review_persists_separate_feedback_record_and_learning_signal(tmp_path) -> None:
    database = tmp_path / "feedback.sqlite3"
    batch = create_batch_from_excel(
        _workbook(
            ["Номер запроса", "Описание", "Услуга", "Пользователь"],
            [["FB-1", "Не работает важная функция в приложении", "Почта", "Оператор"]],
        ),
        "feedback.xlsx",
        database,
    )
    ticket = list_batch_tickets(batch["id"], database)[0]
    save_analysis(ticket["id"], _analysis(), database)
    staged = list_batch_tickets(batch["id"], database)[0]

    assert staged["needs_training_review"] is True
    assert staged["learning_priority"] > 0

    confirm_category(ticket["id"], "Категория B", database)
    confirm_route(ticket["id"], "(2 линия)", database)
    complete_ticket(ticket["id"], database)

    connection = sqlite3.connect(database)
    connection.row_factory = sqlite3.Row
    feedback = connection.execute("SELECT * FROM production_feedback WHERE request_id='FB-1'").fetchone()
    historical_count = connection.execute(
        "SELECT COUNT(*) FROM tickets WHERE source_hash <> 'operator-review'"
    ).fetchone()[0]
    connection.close()

    assert feedback is not None
    assert feedback["model_category"] == "Категория A"
    assert feedback["operator_final_category"] == "Категория B"
    assert feedback["category_corrected"] == 1
    assert feedback["route_corrected"] == 1
    assert feedback["accepted_without_change"] == 0
    assert feedback["model_version"] == "v5.2"
    assert feedback["prediction_timestamp"]
    assert feedback["confirmation_timestamp"]
    assert json.loads(feedback["available_fields_json"])
    assert "component" in json.loads(feedback["missing_fields_json"])
    assert feedback["needs_training_review"] == 1
    assert historical_count == 0


def test_confirm_category_api_recalculates_route_and_resets_route_confirmation(tmp_path, monkeypatch) -> None:
    database = tmp_path / "api-routing.sqlite3"
    batch = create_batch_from_excel(
        _workbook(
            ["Номер запроса", "Описание", "Услуга"],
            [["API-1", "Не работает получение отправления по QR-коду", "Почта"]],
        ),
        "api-routing.xlsx",
        database,
    )
    ticket = list_batch_tickets(batch["id"], database)[0]
    save_analysis(ticket["id"], _analysis(), database)
    confirm_category(ticket["id"], "Категория A", database)
    confirm_route(ticket["id"], "(2 линия)", database)

    recalculated = {
        "label": "(3 линия)",
        "confidence": 0.77,
        "accepted": True,
        "review_required": False,
        "threshold": 0.55,
        "alternatives": [
            {"label": "(3 линия)", "confidence": 0.77},
            {"label": "(2 линия)", "confidence": 0.18},
            {"label": "(1 линия)", "confidence": 0.05},
        ],
        "signals": [],
        "explanation": "",
        "mode": "confirmed_category",
        "category_prior_used": True,
    }

    monkeypatch.setattr(
        routes_module,
        "confirm_category",
        lambda ticket_id, category: confirm_category(ticket_id, category, database),
    )
    monkeypatch.setattr(
        routes_module,
        "ticket_analysis_payload",
        lambda ticket_id: ticket_analysis_payload(ticket_id, database),
    )
    monkeypatch.setattr(
        routes_module,
        "save_routing_recommendation",
        lambda ticket_id, routing: save_routing_recommendation(ticket_id, routing, database),
    )
    monkeypatch.setattr(routes_module, "route_ticket", lambda *args, **kwargs: recalculated)
    monkeypatch.setattr(routes_module, "runtime", lambda: {})
    monkeypatch.setattr(routes_module, "DATABASE_PATH", database)

    response = TestClient(app).post(
        f"/api/incoming/tickets/{ticket['id']}/confirm-category",
        json={"category": "Категория B"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["category_confirmed"] == 1
    assert body["route_confirmed"] == 0
    assert body["operator_route"] is None
    assert body["model_route"] == "(3 линия)"
    assert body["analysis"]["routing"]["mode"] == "confirmed_category"

    connection = sqlite3.connect(database)
    events = [
        row[0]
        for row in connection.execute(
            "SELECT event_type FROM incoming_ticket_audit WHERE incoming_ticket_id=? ORDER BY id",
            (ticket["id"],),
        ).fetchall()
    ]
    connection.close()
    assert events[-2:] == ["category_confirmed", "routing_recalculated"]
