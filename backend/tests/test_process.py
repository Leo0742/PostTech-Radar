from __future__ import annotations

import csv
import sqlite3
from pathlib import Path

from app.services.data_service import ensure_database_schema, participation_flags
from app.services.process_service import import_status_history, process_summary


def test_participation_flags_use_reaction_or_work_time() -> None:
    row = {
        "Суммарное время реакции 1 линии": "00:01:00",
        "Суммарное время работы 1 линии": "00:00:00",
        "Суммарное время реакции 2 линии": "00:00:00",
        "Суммарное время работы 2 линии": "01:00:00",
        "Суммарное время реакции 3 линии": None,
        "Суммарное время работы 3 линии": "00:00:00",
        "Суммарное время реакции 4 линии": "00:00:00",
        "Суммарное время работы 4 линии": "00:00:00",
    }
    assert participation_flags(row) == (1, 1, 0, 0)


def test_status_history_import_and_transition_aggregation(tmp_path: Path) -> None:
    database = tmp_path / "process.db"
    ensure_database_schema(database)
    source = tmp_path / "events.csv"
    with source.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["request_id", "status", "entered_at", "left_at", "source"])
        writer.writeheader()
        writer.writerows(
            [
                {"request_id": "INC-1", "status": "Новый", "entered_at": "2025-01-01T10:00:00", "left_at": "2025-01-01T11:00:00", "source": "test"},
                {"request_id": "INC-1", "status": "В работе", "entered_at": "2025-01-01T11:00:00", "left_at": "2025-01-01T13:00:00", "source": "test"},
                {"request_id": "INC-1", "status": "Закрыт", "entered_at": "2025-01-01T13:00:00", "left_at": "", "source": "test"},
            ]
        )
    result = import_status_history(source, database)
    summary = process_summary(database=database)
    assert result["inserted"] == 3
    assert summary["status_history"]["available"] is True
    assert summary["status_history"]["transitions"] == [
        {"from": "Новый", "to": "В работе", "count": 1, "share": 0.5},
        {"from": "В работе", "to": "Закрыт", "count": 1, "share": 0.5},
    ]
    dwell = {row["status"]: row for row in summary["status_history"]["dwell"]}
    assert dwell["Новый"]["median_seconds"] == 3600
    assert dwell["В работе"]["median_seconds"] == 7200


def test_process_empty_state_is_honest_when_no_events(tmp_path: Path) -> None:
    database = tmp_path / "empty.db"
    ensure_database_schema(database)
    summary = process_summary(database=database)
    assert summary["status_history"]["available"] is False
    assert "невозможно восстановить" in summary["status_history"]["message"]
    assert summary["participation"]["edge_semantics"] == "участие → финальная линия; не хронологический переход"


def test_participation_graph_uses_flags_not_line_number_order(tmp_path: Path) -> None:
    database = tmp_path / "participation.db"
    ensure_database_schema(database)
    connection = sqlite3.connect(database)
    base = (
        "INSERT INTO tickets (request_id, registration_date, service, category, request_type, description, "
        "normalized_description, criticality, urgency, priority, service_class, timezone, status, overdue, "
        "final_line, clarifications_count, line1_participated, line2_participated, line3_participated, "
        "line4_participated, participants_count, result, raw_json, source_hash, updated_at) "
        "VALUES (?, '2025-01-01T00:00:00', 'S', 'C', 'T', 'D', 'd', 'K', 'U', 'P', 'SC', 'TZ', "
        "'Закрыт', ?, ?, 0, ?, ?, ?, 0, ?, 'ok', '{}', 'x', '2026-01-01T00:00:00')"
    )
    connection.execute(base, ("1", 0, "(2 линия)", 1, 1, 0, 2))
    connection.execute(base, ("2", 1, "(3 линия)", 1, 0, 1, 2))
    connection.commit()
    connection.close()
    summary = process_summary(database=database)["participation"]
    assert {row["combination"] for row in summary["combinations"]} == {"1+2", "1+3"}
    assert any(edge == {"participant": "1", "resolver": "2", "count": 1} for edge in summary["resolver_edges"])
