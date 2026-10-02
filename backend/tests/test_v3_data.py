from __future__ import annotations

from datetime import datetime
from pathlib import Path

from app.core.features import REQUIRED_COLUMNS
from app.ml.v3.dataset import DatasetConfig, challenge_dataset_config, training_corpus_summary
from app.services.data_service import import_workbook, load_rows
from openpyxl import Workbook


def _challenge_row(request_id: str, category: str) -> dict[str, object]:
    return {
        "Номер запроса": request_id,
        "Дата регистрации": datetime(2026, 1, 1, 12, 0),
        "Услуга": "Тестовая услуга",
        "Компонент услуги 1 уровня": "Компонент",
        "Тип запроса": "Инцидент",
        "Описание 2": f"Описание {request_id}",
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
        "Результат работ": "Выполнено",
        "Количество уточнений": 0,
    }


def _write_workbook(path: Path, headers: list[str], rows: list[dict[str, object]]) -> None:
    workbook = Workbook()
    sheet = workbook.active
    sheet.append(headers)
    for row in rows:
        sheet.append([row.get(header) for header in headers])
    workbook.save(path)


def test_delta_import_recomputes_top_k_from_complete_database(tmp_path: Path) -> None:
    database = tmp_path / "cumulative.db"
    initial = tmp_path / "initial.xlsx"
    delta = tmp_path / "delta.xlsx"
    _write_workbook(
        initial,
        list(REQUIRED_COLUMNS),
        [_challenge_row("A-1", "A"), _challenge_row("A-2", "A"), _challenge_row("A-3", "A"), _challenge_row("B-1", "B")],
    )
    _write_workbook(
        delta,
        list(REQUIRED_COLUMNS),
        [_challenge_row("C-1", "C"), _challenge_row("C-2", "C"), _challenge_row("C-3", "C"), _challenge_row("C-4", "C")],
    )
    config = challenge_dataset_config(top_k=2, minimum_class_support=1)

    import_workbook(initial, database, mode="upsert", dataset_config=config)
    result = import_workbook(delta, database, mode="delta", dataset_config=config)

    assert result["records"] == 8
    assert result["training_summary"]["eligible_records"] == 8
    assert result["training_summary"]["top_k"] == [
        {"name": "C", "count": 4},
        {"name": "A", "count": 3},
    ]
    assert training_corpus_summary(database, config)["dataset_sha256"] == result["training_summary"]["dataset_sha256"]


def test_generic_column_mapping_imports_same_logical_training_fields(tmp_path: Path) -> None:
    database = tmp_path / "generic.db"
    source = tmp_path / "future_snapshot.xlsx"
    headers = ["Ticket", "Created", "Body", "Kind", "Resolver", "Product"]
    _write_workbook(
        source,
        headers,
        [
            {
                "Ticket": "NEW-42",
                "Created": datetime(2026, 2, 3, 9, 30),
                "Body": "Не формируется QR-код",
                "Kind": "QR",
                "Resolver": "L2",
                "Product": "Мобильное приложение",
            }
        ],
    )
    config = DatasetConfig(
        id_column="Ticket",
        registration_date_column="Created",
        text_column="Body",
        label_column="Kind",
        route_column="Resolver",
        safe_registration_features={"service": "Product"},
        top_k=15,
        minimum_class_support=1,
    )

    result = import_workbook(source, database, mode="snapshot", dataset_config=config)
    row = load_rows(database)[0]

    assert result["training_summary"]["eligible_records"] == 1
    assert row["request_id"] == "NEW-42"
    assert row["description"] == "Не формируется QR-код"
    assert row["category"] == "QR"
    assert row["final_line"] == "L2"
    assert row["service"] == "Мобильное приложение"
