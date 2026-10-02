from __future__ import annotations

import sqlite3
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np

from app.core.config import DATABASE_PATH
from app.services.data_service import ensure_database_schema, human_duration, metadata_value


@dataclass(frozen=True)
class AnalyticsFilters:
    date_from: str | None = None
    date_to: str | None = None
    service: str | None = None
    category: str | None = None
    priority: str | None = None
    support_line: str | None = None


def normalize_date_bounds(date_from: str | None, date_to: str | None) -> tuple[str | None, str | None]:
    start = f"{date.fromisoformat(date_from).isoformat()}T00:00:00" if date_from else None
    end = f"{(date.fromisoformat(date_to) + timedelta(days=1)).isoformat()}T00:00:00" if date_to else None
    return start, end


def _where(filters: AnalyticsFilters) -> tuple[str, list[str]]:
    clauses: list[str] = []
    values: list[str] = []
    start, end = normalize_date_bounds(filters.date_from, filters.date_to)
    if start:
        clauses.append("registration_date >= ?")
        values.append(start)
    if end:
        clauses.append("registration_date < ?")
        values.append(end)
    for field, value in (
        ("service", filters.service), ("category", filters.category),
        ("priority", filters.priority), ("final_line", filters.support_line),
    ):
        if value:
            clauses.append(f"{field} = ?")
            values.append(value)
    return (" WHERE " + " AND ".join(clauses) if clauses else "", values)


def _smoothed(overdue: int, count: int, global_rate: float, prior: int = 30) -> float:
    return (overdue + prior * global_rate) / (count + prior) if count + prior else 0.0


def _breakdown(
    connection: sqlite3.Connection, field: str, where: str, values: list[str], global_rate: float
) -> list[dict[str, Any]]:
    if field not in {"category", "service", "priority", "final_line"}:
        raise ValueError("Недопустимое поле группировки")
    rows = connection.execute(
        f"SELECT {field} AS name, overdue, actual_duration_seconds, clarifications_count FROM tickets{where}", values
    ).fetchall()
    grouped: dict[str, list[sqlite3.Row]] = defaultdict(list)
    for row in rows:
        grouped[row["name"] or "Не указано"].append(row)
    result = []
    for name, group in grouped.items():
        durations = [row["actual_duration_seconds"] for row in group if row["actual_duration_seconds"] is not None]
        count = len(group)
        overdue = sum(row["overdue"] for row in group)
        median = int(np.median(durations)) if durations else 0
        p90 = int(np.percentile(durations, 90)) if durations else 0
        result.append({
            "name": name, "count": count, "sample_n": count, "overdue": overdue,
            "raw_overdue_rate": round(overdue / count, 4), "overdue_share": round(overdue / count, 4),
            "smoothed_risk": round(_smoothed(overdue, count, global_rate), 4),
            "median_duration_seconds": median, "p90_duration_seconds": p90,
            "avg_duration_seconds": int(np.mean(durations)) if durations else 0,
            "avg_clarifications": round(float(np.mean([row["clarifications_count"] for row in group])), 2),
        })
    return sorted(result, key=lambda item: item["count"], reverse=True)


def analytics_summary(filters: AnalyticsFilters, database: Path = DATABASE_PATH) -> dict[str, Any]:
    ensure_database_schema(database)
    connection = sqlite3.connect(database)
    connection.row_factory = sqlite3.Row
    where, values = _where(filters)
    rows = connection.execute(
        "SELECT overdue, actual_duration_seconds, clarifications_count, participants_count FROM tickets" + where, values
    ).fetchall()
    durations = [row["actual_duration_seconds"] for row in rows if row["actual_duration_seconds"] is not None]
    total = len(rows)
    overdue = sum(row["overdue"] for row in rows)
    global_total, global_overdue = connection.execute("SELECT COUNT(*), COALESCE(SUM(overdue), 0) FROM tickets").fetchone()
    global_rate = global_overdue / global_total if global_total else 0.0
    threshold_value = metadata_value("clarification_threshold", database)
    clarification_threshold = int(threshold_value) if threshold_value is not None else 3
    high_clarifications = sum(row["clarifications_count"] >= clarification_threshold for row in rows)
    multi_line = sum(row["participants_count"] > 1 for row in rows)
    time_rows = connection.execute(
        "SELECT substr(registration_date, 1, 7) AS month, COUNT(*) AS count, SUM(overdue) AS overdue "
        "FROM tickets" + where + " GROUP BY month ORDER BY month", values,
    ).fetchall()
    breakdowns = {
        "categories": _breakdown(connection, "category", where, values, global_rate),
        "services": _breakdown(connection, "service", where, values, global_rate),
        "priorities": _breakdown(connection, "priority", where, values, global_rate),
        "support_lines": _breakdown(connection, "final_line", where, values, global_rate),
    }
    connection.close()
    median = int(np.median(durations)) if durations else 0
    p90 = int(np.percentile(durations, 90)) if durations else 0
    return {
        "filters": filters.__dict__,
        "kpis": {
            "tickets": total, "overdue": overdue, "overdue_share": round(overdue / total, 4) if total else 0,
            "smoothed_risk": round(_smoothed(overdue, total, global_rate), 4),
            "median_duration_seconds": median, "median_duration": human_duration(median),
            "p90_duration_seconds": p90, "p90_duration": human_duration(p90), "multi_line": multi_line,
            "high_clarifications": high_clarifications, "clarification_threshold": clarification_threshold,
        },
        "breakdowns": breakdowns,
        "time": [{
            "month": row["month"], "count": row["count"], "overdue": row["overdue"] or 0,
            "overdue_share": round((row["overdue"] or 0) / row["count"], 4),
            "smoothed_risk": round(_smoothed(row["overdue"] or 0, row["count"], global_rate), 4),
        } for row in time_rows],
        "methodology": {
            "sla": "Авторитетная метка — Просрочен?*; длительность — исходное поле SLA без 24-часового ограничения.",
            "risk": "Риск ранжируется по Beta-сглаженной доле; raw rate и размер выборки показаны отдельно.",
            "multi_line": "Линия участвует, если время реакции или работы больше нуля; порядок линий не подразумевается.",
            "many_clarifications": f"Глобальный порог полного набора: количество уточнений ≥ {clarification_threshold}; фильтры его не меняют.",
            "dates": "Дата начала включительна; дата окончания включает весь выбранный день.",
        },
    }
