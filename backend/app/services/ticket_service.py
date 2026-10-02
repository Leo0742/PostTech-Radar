from __future__ import annotations

import json
import math
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.core.config import DATABASE_PATH, MODELS_DIR
from app.services.analytics_service import normalize_date_bounds
from app.services.data_service import ensure_database_schema, human_duration, normalize_description

EDITABLE_TICKET_FIELDS = (
    "registration_date", "user_name", "service", "component", "category", "request_type", "description",
    "criticality", "urgency", "priority", "service_class", "timezone", "final_line", "result", "overdue",
    "actual_duration_seconds", "clarifications_count",
)
NON_NULL_TICKET_FIELDS = {
    "registration_date", "service", "category", "request_type", "description", "criticality", "urgency",
    "priority", "service_class", "timezone", "final_line", "result", "overdue", "clarifications_count",
}


def _connection(database: Path = DATABASE_PATH) -> sqlite3.Connection:
    ensure_database_schema(database)
    connection = sqlite3.connect(database)
    connection.row_factory = sqlite3.Row
    return connection


def _snapshot(row: sqlite3.Row | dict[str, Any]) -> dict[str, Any]:
    values = dict(row)
    return {field: (bool(values[field]) if field == "overdue" else values.get(field)) for field in EDITABLE_TICKET_FIELDS}


def ticket_detail(request_id: str, database: Path = DATABASE_PATH) -> dict[str, Any] | None:
    connection = _connection(database)
    row = connection.execute("SELECT * FROM tickets WHERE request_id=?", (request_id,)).fetchone()
    connection.close()
    if row is None:
        return None
    result = dict(row)
    result["overdue"] = bool(result["overdue"])
    result["actual_duration"] = human_duration(result["actual_duration_seconds"])
    result["raw"] = json.loads(result.pop("raw_json"))
    return result


def list_ticket_revisions(request_id: str, database: Path = DATABASE_PATH) -> list[dict[str, Any]]:
    connection = _connection(database)
    rows = connection.execute(
        "SELECT revision_id, request_id, changed_at, changed_by, changed_fields_json, before_json, after_json "
        "FROM ticket_revisions WHERE request_id=? ORDER BY revision_id DESC",
        (request_id,),
    ).fetchall()
    connection.close()
    return [
        {
            "revision_id": row["revision_id"],
            "request_id": row["request_id"],
            "changed_at": row["changed_at"],
            "changed_by": row["changed_by"],
            "changed_fields": json.loads(row["changed_fields_json"]),
            "before": json.loads(row["before_json"]),
            "after": json.loads(row["after_json"]),
        }
        for row in rows
    ]


def update_ticket(
    request_id: str,
    payload: dict[str, Any],
    *,
    changed_by: str = "Оператор",
    database: Path = DATABASE_PATH,
) -> dict[str, Any] | None:
    connection = _connection(database)
    row = connection.execute("SELECT * FROM tickets WHERE request_id=?", (request_id,)).fetchone()
    if row is None:
        connection.close()
        return None

    before = _snapshot(row)
    normalized: dict[str, Any] = {}
    for field, value in payload.items():
        if field not in EDITABLE_TICKET_FIELDS:
            continue
        if field in NON_NULL_TICKET_FIELDS and value is None:
            connection.close()
            raise ValueError(f"Поле {field} не может быть пустым")
        if field == "overdue" and value is not None:
            normalized[field] = int(bool(value))
        else:
            normalized[field] = value

    changed_fields = [
        field for field, value in normalized.items()
        if (bool(row[field]) if field == "overdue" else row[field]) != (bool(value) if field == "overdue" else value)
    ]
    if not changed_fields:
        connection.close()
        return ticket_detail(request_id, database)

    now = datetime.now(UTC).isoformat()
    assignments = [f"{field}=?" for field in changed_fields]
    values = [normalized[field] for field in changed_fields]
    if "description" in changed_fields:
        assignments.append("normalized_description=?")
        values.append(normalize_description(str(normalized["description"])))
    assignments.append("updated_at=?")
    values.append(now)

    connection.execute("BEGIN")
    try:
        connection.execute(
            f"UPDATE tickets SET {', '.join(assignments)} WHERE request_id=?",
            [*values, request_id],
        )
        refreshed = connection.execute("SELECT * FROM tickets WHERE request_id=?", (request_id,)).fetchone()
        if refreshed is None:
            raise RuntimeError("Обращение исчезло во время обновления")
        after = _snapshot(refreshed)
        connection.execute(
            "INSERT INTO ticket_revisions "
            "(request_id, changed_at, changed_by, changed_fields_json, before_json, after_json) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                request_id,
                now,
                (changed_by or "Оператор").strip()[:100] or "Оператор",
                json.dumps(changed_fields, ensure_ascii=False),
                json.dumps(before, ensure_ascii=False, sort_keys=True),
                json.dumps(after, ensure_ascii=False, sort_keys=True),
            ),
        )
        connection.commit()
    except Exception:
        connection.rollback()
        connection.close()
        raise
    connection.close()
    return ticket_detail(request_id, database)


def list_tickets(
    *,
    page: int = 1,
    page_size: int = 20,
    query: str | None = None,
    service: str | None = None,
    category: str | None = None,
    priority: str | None = None,
    support_line: str | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    database: Path = DATABASE_PATH,
) -> dict[str, Any]:
    clauses: list[str] = []
    values: list[Any] = []
    if query:
        clauses.append("(CAST(request_id AS TEXT) LIKE ? OR description LIKE ?)")
        pattern = f"%{query.strip()}%"
        values.extend([pattern, pattern])
    for field, value in (
        ("service", service),
        ("category", category),
        ("priority", priority),
        ("final_line", support_line),
    ):
        if value:
            clauses.append(f"{field} = ?")
            values.append(value)
    start, end = normalize_date_bounds(date_from, date_to)
    if start:
        clauses.append("registration_date >= ?")
        values.append(start)
    if end:
        clauses.append("registration_date < ?")
        values.append(end)
    where = " WHERE " + " AND ".join(clauses) if clauses else ""
    connection = _connection(database)
    total = connection.execute("SELECT COUNT(*) FROM tickets" + where, values).fetchone()[0]
    rows = connection.execute(
        "SELECT request_id, registration_date, service, category, request_type, description, priority, final_line, overdue, "
        "actual_duration_seconds, clarifications_count FROM tickets"
        + where
        + " ORDER BY registration_date DESC LIMIT ? OFFSET ?",
        [*values, page_size, (page - 1) * page_size],
    ).fetchall()
    connection.close()
    items = []
    for row in rows:
        item = dict(row)
        item["overdue"] = bool(item["overdue"])
        item["duration"] = human_duration(item.pop("actual_duration_seconds"))
        items.append(item)
    return {
        "items": items,
        "total": total,
        "page": page,
        "page_size": page_size,
        "total_pages": math.ceil(total / page_size) if total else 0,
    }


def dataset_options(database: Path = DATABASE_PATH) -> dict[str, list[str]]:
    connection = _connection(database)
    mapping = {
        "services": "service",
        "components": "component",
        "request_types": "request_type",
        "criticalities": "criticality",
        "urgencies": "urgency",
        "priorities": "priority",
        "service_classes": "service_class",
        "timezones": "timezone",
        "categories": "category",
        "support_lines": "final_line",
    }
    result: dict[str, list[str]] = {}
    for key, field in mapping.items():
        rows = connection.execute(
            f"SELECT DISTINCT {field} FROM tickets WHERE {field} IS NOT NULL AND {field} != '' ORDER BY {field}"
        ).fetchall()
        result[key] = [row[0] for row in rows]
    connection.close()

    # Training/evaluation batches are deliberately not inserted into the
    # historical production database.  Keep the operator category selector in
    # sync with the promoted model taxonomy by merging its explicit labels.
    taxonomy_path = MODELS_DIR / "v5" / "category_lite_v3.json"
    if taxonomy_path.exists():
        try:
            taxonomy = json.loads(taxonomy_path.read_text(encoding="utf-8"))
            model_categories = [str(value) for value in taxonomy.get("label_names", []) if str(value).strip()]
            if model_categories:
                result["categories"] = sorted(set(result["categories"]) | set(model_categories))
        except (OSError, ValueError, TypeError):
            pass
    return result
