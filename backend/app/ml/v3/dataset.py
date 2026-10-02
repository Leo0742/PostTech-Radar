from __future__ import annotations

import hashlib
import json
import sqlite3
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.core.features import REQUIRED_COLUMNS


@dataclass(frozen=True)
class DatasetConfig:
    id_column: str
    registration_date_column: str
    text_column: str
    label_column: str
    route_column: str
    safe_registration_features: Mapping[str, str] = field(default_factory=dict)
    operational_columns: Mapping[str, str] = field(default_factory=dict)
    required_columns: tuple[str, ...] = ()
    top_k: int = 15
    minimum_class_support: int = 1

    def __post_init__(self) -> None:
        if self.top_k < 1:
            raise ValueError("top_k must be at least 1")
        if self.minimum_class_support < 1:
            raise ValueError("minimum_class_support must be at least 1")

    @property
    def essential_columns(self) -> tuple[str, ...]:
        values = (
            self.id_column,
            self.registration_date_column,
            self.text_column,
            self.label_column,
            self.route_column,
            *self.safe_registration_features.values(),
        )
        return tuple(dict.fromkeys(values))

    def column_for(self, logical_name: str) -> str | None:
        fixed = {
            "request_id": self.id_column,
            "registration_date": self.registration_date_column,
            "description": self.text_column,
            "category": self.label_column,
            "final_line": self.route_column,
        }
        return fixed.get(logical_name) or self.safe_registration_features.get(logical_name) or self.operational_columns.get(logical_name)


def challenge_dataset_config(*, top_k: int = 15, minimum_class_support: int = 1) -> DatasetConfig:
    return DatasetConfig(
        id_column="Номер запроса",
        registration_date_column="Дата регистрации",
        text_column="Описание 2",
        label_column="Вид запроса",
        route_column="Кем решен (группа)",
        safe_registration_features={
            "service": "Услуга",
            "component": "Компонент услуги 1 уровня",
            "request_type": "Тип запроса",
            "criticality": "Критичность",
            "urgency": "Срочность",
            "priority": "Приоритет",
            "service_class": "Класс обслуживания",
            "timezone": "Часовой пояс запроса",
        },
        operational_columns={
            "status": "Статус",
            "overdue": "Просрочен?*",
            "actual_duration": "Фактическая длительность выполнения запроса (SLA)",
            "clarifications_count": "Количество уточнений",
            "clarification_duration": "Суммарное время уточнений",
            "result": "Результат работ",
            "line1_reaction": "Суммарное время реакции 1 линии",
            "line1_work": "Суммарное время работы 1 линии",
            "line2_reaction": "Суммарное время реакции 2 линии",
            "line2_work": "Суммарное время работы 2 линии",
            "line3_reaction": "Суммарное время реакции 3 линии",
            "line3_work": "Суммарное время работы 3 линии",
            "line4_reaction": "Суммарное время реакции 4 линии",
            "line4_work": "Суммарное время работы 4 линии",
        },
        required_columns=REQUIRED_COLUMNS,
        top_k=top_k,
        minimum_class_support=minimum_class_support,
    )


def source_value(row: Mapping[str, Any], config: DatasetConfig, logical_name: str, default: Any = "") -> Any:
    column = config.column_for(logical_name)
    return row.get(column, default) if column else default


def training_corpus_summary(database: Path, config: DatasetConfig | None = None) -> dict[str, Any]:
    config = config or challenge_dataset_config()
    connection = sqlite3.connect(database)
    connection.row_factory = sqlite3.Row
    rows = [dict(row) for row in connection.execute(
        "SELECT request_id, registration_date, service, component, category, request_type, description, "
        "normalized_description, criticality, urgency, priority, service_class, timezone, final_line "
        "FROM tickets ORDER BY request_id"
    )]
    connection.close()
    labeled = [row for row in rows if str(row["category"] or "").strip() and str(row["description"] or "").strip()]
    counts = Counter(str(row["category"]) for row in labeled)
    supported = {label: count for label, count in counts.items() if count >= config.minimum_class_support}
    ranked = sorted(supported.items(), key=lambda item: (-item[1], item[0]))
    top = [{"name": label, "count": count} for label, count in ranked[: config.top_k]]
    eligible_labels = set(supported)
    eligible = [row for row in labeled if row["category"] in eligible_labels]
    canonical = json.dumps(eligible, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    route_counts = Counter(str(row["final_line"] or "") for row in eligible)
    return {
        "records": len(rows),
        "database_records": len(rows),
        "labeled_records": len(labeled),
        "eligible_records": len(eligible),
        "distinct_categories": len(counts),
        "supported_categories": len(supported),
        "minimum_class_support": config.minimum_class_support,
        "configured_top_k": config.top_k,
        "top_k": top,
        "top15": top,
        "top15_records": sum(item["count"] for item in top),
        "class_counts": [{"name": label, "count": count} for label, count in ranked],
        "support_lines": dict(sorted(route_counts.items())),
        "dataset_sha256": digest,
    }


def eligible_training_rows(database: Path, config: DatasetConfig | None = None) -> list[dict[str, Any]]:
    config = config or challenge_dataset_config()
    summary = training_corpus_summary(database, config)
    supported = {item["name"] for item in summary["class_counts"]}
    connection = sqlite3.connect(database)
    connection.row_factory = sqlite3.Row
    rows = [dict(row) for row in connection.execute("SELECT * FROM tickets ORDER BY request_id")]
    connection.close()
    return [row for row in rows if row["category"] in supported and str(row["description"] or "").strip()]

