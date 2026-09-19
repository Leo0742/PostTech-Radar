from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.v5_data_audit import read_xlsx_rows


DEFAULT_DATASET = PROJECT_ROOT / "data" / "raw" / "Обращения_1931.xlsx"

REGISTRATION_FIELDS = (
    "registration_date",
    "user",
    "service",
    "component",
    "request_type",
    "description",
    "criticality",
    "urgency",
    "priority",
    "service_class",
    "timezone",
)

SOURCE_TO_INTERNAL = {
    "Пользователь": "user",
    "Услуга": "service",
    "Компонент услуги 1 уровня": "component",
    "Тип запроса": "request_type",
    "Описание 2": "description",
    "Критичность": "criticality",
    "Срочность": "urgency",
    "Приоритет": "priority",
    "Класс обслуживания": "service_class",
    "Часовой пояс запроса": "timezone",
}


def _request_id(value: Any) -> str:
    if value in {None, ""}:
        raise ValueError("Номер запроса cannot be empty")
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value).strip()


def _registration_datetime(value: Any) -> str:
    if value in {None, ""}:
        raise ValueError("Дата регистрации cannot be empty")
    if isinstance(value, (int, float)):
        moment = datetime(1899, 12, 30) + timedelta(days=float(value))
        return moment.isoformat(timespec="microseconds")
    text = str(value).strip()
    moment = datetime.fromisoformat(text.replace("Z", "+00:00"))
    return moment.isoformat(timespec="microseconds")


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = " ".join(str(value).split())
    return text or None


def load_v5_rows(path: Path = DEFAULT_DATASET) -> list[dict[str, Any]]:
    _sheet, _headers, source_rows, _storage = read_xlsx_rows(Path(path))
    rows: list[dict[str, Any]] = []
    for source in source_rows:
        category = _optional_text(source.get("Вид запроса"))
        routing_target = _optional_text(source.get("Кем решен (группа)"))
        if category is None:
            raise ValueError("Вид запроса cannot be empty")
        if routing_target is None:
            raise ValueError("Кем решен (группа) cannot be empty")

        row: dict[str, Any] = {
            "request_id": _request_id(source.get("Номер запроса")),
            "category": category,
            "routing_target": routing_target,
            "registration_date": _registration_datetime(source.get("Дата регистрации")),
        }
        for source_name, internal_name in SOURCE_TO_INTERNAL.items():
            row[internal_name] = _optional_text(source.get(source_name))
        rows.append(row)

    if len({row["request_id"] for row in rows}) != len(rows):
        raise ValueError("request_id must be unique")
    return rows


def safe_registration_view(row: Mapping[str, Any]) -> dict[str, Any]:
    return {field: row.get(field) for field in REGISTRATION_FIELDS}
