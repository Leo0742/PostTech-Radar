from __future__ import annotations

from pathlib import Path

from app.services.data_service import inspect_workbook

DATASET = Path(__file__).parents[2] / "data" / "raw" / "Обращения_1931.xlsx"


def test_real_dataset_integrity() -> None:
    summary = inspect_workbook(DATASET)
    assert summary["records"] == 1931
    assert summary["columns"] == 34
    assert summary["distinct_categories"] == 43
    assert summary["top15_records"] == 1511
    assert summary["support_lines"] == {
        "(1 линия)": 1192,
        "(2 линия)": 566,
        "(3 линия)": 172,
        "(4 линия)": 1,
    }
    assert summary["sla_labels"]["Просрочен"] == 25
    assert summary["statuses"] == {"Закрыт": 1931}
    assert summary["missing"]["Компонент услуги 1 уровня"] == 619
    assert summary["missing"]["Фактическая длительность выполнения запроса (SLA)"] == 1
    assert summary["top15"][0] == {"name": "Прочее", "count": 316}

