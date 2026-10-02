from __future__ import annotations

import sqlite3

from app.api import routes as routes_module
from app.main import app
from app.services.data_service import ensure_database_schema
from app.services.ticket_service import (
    list_ticket_revisions as service_list_ticket_revisions,
    ticket_detail as service_ticket_detail,
    update_ticket as service_update_ticket,
)
from fastapi.testclient import TestClient


def _seed_ticket(database) -> None:  # noqa: ANN001
    ensure_database_schema(database)
    connection = sqlite3.connect(database)
    connection.execute(
        """
        INSERT INTO tickets (
            request_id, registration_date, user_name, service, component, category, request_type,
            description, normalized_description, criticality, urgency, priority, service_class,
            timezone, status, overdue, final_line, actual_duration_seconds, clarifications_count,
            participants_count, result, raw_json, source_hash, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "EDIT-1", "2026-09-18T10:00:00", "Initial user", "Service A", None, "Category A",
            "Incident", "Original description", "original description", "Low", "Low", "Low",
            "Standard", "Europe/Moscow", "Closed", 0, "(1 линия)", 600, 1, 1,
            "Resolved", "{}", "test", "2026-09-18T10:00:00+00:00",
        ),
    )
    connection.commit()
    connection.close()


def _use_database(monkeypatch, database) -> None:  # noqa: ANN001
    monkeypatch.setattr(
        routes_module,
        "ticket_detail",
        lambda request_id: service_ticket_detail(request_id, database),
    )
    monkeypatch.setattr(
        routes_module,
        "list_ticket_revisions",
        lambda request_id: service_list_ticket_revisions(request_id, database),
    )
    monkeypatch.setattr(
        routes_module,
        "update_ticket",
        lambda request_id, payload, changed_by="Оператор": service_update_ticket(
            request_id,
            payload,
            changed_by=changed_by,
            database=database,
        ),
    )


def test_patch_confirmed_ticket_persists_and_creates_revision(tmp_path, monkeypatch) -> None:  # noqa: ANN001
    database = tmp_path / "confirmed-edit.sqlite3"
    _seed_ticket(database)
    _use_database(monkeypatch, database)
    client = TestClient(app)

    response = client.patch("/api/tickets/EDIT-1", json={"user_name": "Updated user"})

    assert response.status_code == 200
    assert response.json()["user_name"] == "Updated user"

    revisions = client.get("/api/tickets/EDIT-1/revisions")
    assert revisions.status_code == 200
    assert len(revisions.json()) == 1
    assert revisions.json()[0]["changed_fields"] == ["user_name"]
    assert revisions.json()[0]["changed_by"] == "Оператор"
    assert revisions.json()[0]["before"]["user_name"] == "Initial user"
    assert revisions.json()[0]["after"]["user_name"] == "Updated user"

    connection = sqlite3.connect(database)
    persisted = connection.execute(
        "SELECT user_name FROM tickets WHERE request_id='EDIT-1'"
    ).fetchone()[0]
    connection.close()
    assert persisted == "Updated user"


def test_patch_confirmed_ticket_rejects_request_id_change(tmp_path, monkeypatch) -> None:  # noqa: ANN001
    database = tmp_path / "confirmed-edit.sqlite3"
    _seed_ticket(database)
    _use_database(monkeypatch, database)

    response = TestClient(app).patch(
        "/api/tickets/EDIT-1",
        json={"request_id": "EDIT-2", "user_name": "Updated user"},
    )

    assert response.status_code == 422
    assert service_ticket_detail("EDIT-1", database) is not None
    assert service_ticket_detail("EDIT-2", database) is None
    assert service_list_ticket_revisions("EDIT-1", database) == []


def test_patch_confirmed_ticket_noop_does_not_create_revision(tmp_path, monkeypatch) -> None:  # noqa: ANN001
    database = tmp_path / "confirmed-edit.sqlite3"
    _seed_ticket(database)
    _use_database(monkeypatch, database)

    response = TestClient(app).patch("/api/tickets/EDIT-1", json={"priority": "Low"})

    assert response.status_code == 200
    assert response.json()["priority"] == "Low"
    assert service_list_ticket_revisions("EDIT-1", database) == []
