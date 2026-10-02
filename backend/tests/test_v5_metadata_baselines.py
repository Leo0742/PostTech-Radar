from __future__ import annotations

from pathlib import Path

from scripts.v5_metadata_baselines import build_metadata_row, load_source_rows, run_view


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET = PROJECT_ROOT / "data" / "raw" / "Обращения_1931.xlsx"
PROTOCOL = PROJECT_ROOT / "artifacts" / "gpu_research_v5" / "protocol" / "protocol.json"
CONTRACT = PROJECT_ROOT / "artifacts" / "gpu_research_v5" / "data_contract.json"


def test_metadata_row_is_missing_aware_and_uses_registration_time_calendar_only() -> None:
    source = {
        "Дата регистрации": 45660.75,
        "Пользователь": "operator",
        "Услуга": "service",
        "Компонент услуги 1 уровня": None,
        "Тип запроса": "Инцидент",
        "Критичность": "Средняя",
        "Срочность": "Низкая",
        "Приоритет": "(3) Средний",
        "Класс обслуживания": "7x24",
        "Часовой пояс запроса": "Москва",
    }

    values = build_metadata_row(source)

    assert values["component"] == "__MISSING__"
    assert values["service"] == "service"
    assert values["registration_month"].isdigit()
    assert values["registration_weekday"].isdigit()
    assert values["registration_hour"].isdigit()
    assert "description" not in values
    assert "Номер запроса" not in values


def test_metadata_top15_baseline_never_evaluates_internal_lockbox() -> None:
    rows = load_source_rows(DATASET)
    payload = run_view(rows, PROTOCOL, CONTRACT, view="top15", quick=True)

    lockbox_ids = set(payload["protocol"]["internal_lockbox_request_ids"])
    evaluated_ids = {item["request_id"] for item in payload["oof_predictions_repeat0"]}
    assert lockbox_ids.isdisjoint(evaluated_ids)
    assert payload["real_only_evaluation"] is True
    assert payload["synthetic_rows_in_validation"] == 0
    assert payload["metrics"]["top1_accuracy"] >= 0.0
    assert payload["metrics"]["top3_accuracy"] >= payload["metrics"]["top1_accuracy"]


def test_metadata_full43_oof_probabilities_are_normalized() -> None:
    rows = load_source_rows(DATASET)
    payload = run_view(rows, PROTOCOL, CONTRACT, view="full43", quick=True)

    assert payload["oof_predictions_repeat0"]
    for item in payload["oof_predictions_repeat0"][:25]:
        assert abs(sum(item["probabilities"].values()) - 1.0) < 1e-6
