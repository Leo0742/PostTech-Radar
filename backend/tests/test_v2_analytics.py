from __future__ import annotations

import sqlite3
from pathlib import Path

from app.ml.runtime import historical_sla_risk
from app.services.data_service import ensure_database_schema
from app.services.ticket_service import list_tickets


def _ticket(connection: sqlite3.Connection, request_id: str, day: int, *, overdue: int = 0, category: str = "C") -> None:
    connection.execute(
        """INSERT INTO tickets (
            request_id, registration_date, service, category, request_type, description, normalized_description,
            criticality, urgency, priority, service_class, timezone, status, overdue, final_line,
            clarifications_count, line1_participated, line2_participated, line3_participated, line4_participated,
            participants_count, result, raw_json, source_hash, updated_at
        ) VALUES (?, ?, 'S', ?, 'T', ?, ?, 'K', 'U', 'P', 'SC', 'TZ', 'Закрыт', ?, '(1 линия)',
                  0, 1, 0, 0, 0, 1, 'ok', '{}', 'x', '2026-01-01T00:00:00')""",
        (request_id, f"2025-01-{day:02d}T12:00:00", category, request_id, request_id.lower(), overdue),
    )


def test_sla_risk_smooths_tiny_samples_and_reports_basis(tmp_path: Path) -> None:
    database = tmp_path / "risk.db"
    ensure_database_schema(database)
    connection = sqlite3.connect(database)
    _ticket(connection, "tiny", 1, overdue=1, category="Tiny")
    for index in range(1, 101):
        _ticket(connection, f"normal-{index}", (index % 27) + 1, overdue=int(index <= 5), category="Normal")
    connection.commit()
    connection.close()
    result = historical_sla_risk({}, "Tiny", "(1 линия)", [], database)
    assert result["risk"] < 0.2
    assert result["basis"] == "общий уровень"
    assert result["support_n"] == 101
    assert result["raw_rate"] < 0.1


def test_ticket_listing_has_real_pagination_and_all_filters(tmp_path: Path) -> None:
    database = tmp_path / "pages.db"
    ensure_database_schema(database)
    connection = sqlite3.connect(database)
    for index in range(1, 6):
        _ticket(connection, f"INC-{index}", index, category="Wanted" if index <= 3 else "Other")
    connection.commit()
    connection.close()
    page = list_tickets(page=2, page_size=2, database=database)
    assert page["total"] == 5
    assert page["total_pages"] == 3
    assert page["page"] == 2
    assert len(page["items"]) == 2
    filtered = list_tickets(
        page=1, page_size=20, query="INC", service="S", category="Wanted", priority="P",
        support_line="(1 линия)", date_from="2025-01-01", date_to="2025-01-03", database=database,
    )
    assert filtered["total"] == 3
