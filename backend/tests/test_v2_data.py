from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path

from app.core.features import REQUIRED_COLUMNS
from app.services.analytics_service import AnalyticsFilters, analytics_summary, normalize_date_bounds
from app.services.data_service import ensure_database_schema, import_workbook
from app.services.ticket_service import ticket_detail
from openpyxl import Workbook


def _workbook(path: Path, rows: list[dict[str, object]]) -> None:
    workbook = Workbook()
    sheet = workbook.active
    sheet.append(list(REQUIRED_COLUMNS))
    for row in rows:
        sheet.append([row.get(column) for column in REQUIRED_COLUMNS])
    workbook.save(path)


def _row(request_id: str, description: str, *, category: str = "Прочее") -> dict[str, object]:
    return {
        "Номер запроса": request_id,
        "Дата регистрации": datetime(2025, 1, 3, 12, 0),
        "Услуга": "Тест",
        "Компонент услуги 1 уровня": None,
        "Тип запроса": "Инцидент",
        "Описание 2": description,
        "Критичность": "Средняя",
        "Срочность": "Средняя",
        "Приоритет": "Средний",
        "Класс обслуживания": "Стандарт",
        "Часовой пояс запроса": "MSK",
        "Вид запроса": category,
        "Кем решен (группа)": "(1 линия)",
        "Статус": "Закрыт",
        "Фактическая длительность выполнения запроса (SLA)": "01:00:00",
        "Просрочен?*": "Не просрочен",
        "Суммарное время уточнений": "00:00:00",
        "Результат работ": "Готово",
        "Количество уточнений": 0,
    }


def _insert_ticket(connection: sqlite3.Connection, request_id: str, registration_date: str, **overrides: object) -> None:
    values = {
        "request_id": request_id,
        "registration_date": registration_date,
        "service": "S",
        "component": None,
        "category": "C",
        "request_type": "T",
        "description": request_id,
        "normalized_description": request_id.lower(),
        "criticality": "K",
        "urgency": "U",
        "priority": "P",
        "service_class": "SC",
        "timezone": "TZ",
        "status": "Закрыт",
        "overdue": 0,
        "final_line": "(1 линия)",
        "actual_duration_seconds": 60,
        "clarifications_count": 0,
        "clarification_seconds": 0,
        "line1_participated": 1,
        "line2_participated": 0,
        "line3_participated": 0,
        "line4_participated": 0,
        "participants_count": 1,
        "result": "ok",
        "raw_json": "{}",
        "source_hash": "test",
        "updated_at": "2026-09-16T00:00:00",
    }
    values.update(overrides)
    columns = list(values)
    connection.execute(
        f"INSERT INTO tickets ({', '.join(columns)}) VALUES ({', '.join('?' for _ in columns)})",
        [values[column] for column in columns],
    )


def test_date_bounds_are_inclusive_for_the_whole_date_to_day(tmp_path: Path) -> None:
    database = tmp_path / "dates.db"
    ensure_database_schema(database)
    connection = sqlite3.connect(database)
    for suffix, timestamp in (
        ("midnight", "2025-01-03T00:00:00"),
        ("noon", "2025-01-03T12:00:00"),
        ("last", "2025-01-03T23:59:59"),
        ("next", "2025-01-04T00:00:00"),
        ("previous", "2025-01-02T23:59:59"),
    ):
        _insert_ticket(connection, suffix, timestamp)
    connection.commit()
    connection.close()

    assert normalize_date_bounds("2025-01-03", "2025-01-03") == (
        "2025-01-03T00:00:00",
        "2025-01-04T00:00:00",
    )
    summary = analytics_summary(AnalyticsFilters(date_from="2025-01-03", date_to="2025-01-03"), database)
    assert summary["kpis"]["tickets"] == 3


def test_date_bounds_work_with_other_filters(tmp_path: Path) -> None:
    database = tmp_path / "combined.db"
    ensure_database_schema(database)
    connection = sqlite3.connect(database)
    _insert_ticket(connection, "keep", "2025-01-03T12:00:00", service="Wanted", category="Cat", priority="High", final_line="(2 линия)")
    _insert_ticket(connection, "wrong-service", "2025-01-03T12:00:00", service="Other", category="Cat", priority="High", final_line="(2 линия)")
    connection.commit()
    connection.close()
    filters = AnalyticsFilters("2025-01-03", "2025-01-03", "Wanted", "Cat", "High", "(2 линия)")
    assert analytics_summary(filters, database)["kpis"]["tickets"] == 1


def test_request_ids_are_text_and_support_alphanumeric_values(tmp_path: Path) -> None:
    database = tmp_path / "ids.db"
    ensure_database_schema(database)
    connection = sqlite3.connect(database)
    _insert_ticket(connection, "INC-123456", "2025-01-03T12:00:00")
    connection.commit()
    connection.close()
    assert ticket_detail("INC-123456", database)["request_id"] == "INC-123456"
    schema = sqlite3.connect(database).execute("PRAGMA table_info(tickets)").fetchall()
    assert next(row for row in schema if row[1] == "request_id")[2] == "TEXT"


def test_import_upserts_without_deleting_existing_tickets(tmp_path: Path) -> None:
    database = tmp_path / "upsert.db"
    first = tmp_path / "first.xlsx"
    second = tmp_path / "second.xlsx"
    _workbook(first, [_row("INC-1", "Первая версия")])
    _workbook(second, [_row("INC-1", "Обновлённая версия"), _row("SD-2", "Новая запись")])

    one = import_workbook(first, database, mode="upsert")
    two = import_workbook(second, database, mode="upsert")

    assert one["inserted"] == 1
    assert two == {**two, "inserted": 1, "updated": 1, "unchanged": 0}
    assert ticket_detail("INC-1", database)["description"] == "Обновлённая версия"
    assert ticket_detail("SD-2", database) is not None


def test_clarification_threshold_is_global_and_stable_under_filters(tmp_path: Path) -> None:
    database = tmp_path / "threshold.db"
    ensure_database_schema(database)
    connection = sqlite3.connect(database)
    for index in range(10):
        _insert_ticket(
            connection,
            str(index),
            f"2025-01-{index + 1:02d}T12:00:00",
            category="A" if index < 5 else "B",
            clarifications_count=index,
        )
    connection.execute("INSERT OR REPLACE INTO metadata VALUES ('clarification_threshold', '8')")
    connection.commit()
    connection.close()
    overall = analytics_summary(AnalyticsFilters(), database)
    filtered = analytics_summary(AnalyticsFilters(category="A"), database)
    assert overall["kpis"]["clarification_threshold"] == 8
    assert filtered["kpis"]["clarification_threshold"] == 8
