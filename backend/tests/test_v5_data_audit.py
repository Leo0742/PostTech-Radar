from __future__ import annotations

from pathlib import Path

from scripts.v5_data_audit import build_data_contract, read_xlsx_rows


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET = PROJECT_ROOT / "data" / "raw" / "Обращения_1931.xlsx"


def test_stdlib_xlsx_reader_recovers_real_dataset_shape() -> None:
    sheet_name, headers, rows, _storage = read_xlsx_rows(DATASET)

    assert sheet_name == "Sheet0"
    assert len(headers) == 34
    assert len(rows) == 1931
    assert headers[0] == "Номер запроса"
    assert headers[-1] == "Количество уточнений"


def test_contract_separates_targets_registration_fields_and_leakage() -> None:
    contract = build_data_contract(DATASET)
    by_name = {item["name"]: item for item in contract["fields"]}

    assert contract["targets"] == {
        "category": "Вид запроса",
        "routing": "Кем решен (группа)",
    }
    assert by_name["Номер запроса"]["predictive_role"] == "identity_only"
    assert by_name["Пользователь"]["available_at_registration"] == "yes"
    assert by_name["Пользователь"]["safe_for_category"] is True
    assert by_name["Описание 2"]["safe_for_category"] is True
    assert by_name["Вид запроса"]["is_target"] is True
    assert by_name["Вид запроса"]["safe_for_category"] is False
    assert by_name["Вид запроса"]["routing_usage"] == "confirmed_only_or_oof_prediction"
    assert by_name["Кем решен (группа)"]["is_target"] is True
    assert by_name["Кем решен (группа)"]["safe_for_routing"] is False

    for column in (
        "Статус",
        "Фактическое время выполнения",
        "Фактическая длительность выполнения запроса (SLA)",
        "Просрочен?*",
        "Результат работ",
        "Количество уточнений",
    ):
        assert by_name[column]["post_resolution_or_leakage"] is True
        assert by_name[column]["safe_for_category"] is False
        assert by_name[column]["safe_for_routing"] is False


def test_uncertain_sla_fields_are_not_core_model_features() -> None:
    contract = build_data_contract(DATASET)
    by_name = {item["name"]: item for item in contract["fields"]}

    for column in (
        "Норматив ное время обработки (SLA)",
        "Плановое время выполнения",
        "Крайний срок обработки",
    ):
        assert by_name[column]["available_at_registration"] == "uncertain"
        assert by_name[column]["optional"] is True
        assert by_name[column]["safe_for_category"] is False
        assert by_name[column]["safe_for_routing"] is False
        assert by_name[column]["predictive_role"] == "conditional_optional_only"


def test_contract_has_no_forbidden_core_features() -> None:
    contract = build_data_contract(DATASET)
    forbidden = {
        item["name"]
        for item in contract["fields"]
        if item["post_resolution_or_leakage"] or item["is_target"]
    }

    assert forbidden.isdisjoint(contract["core_category_features"])
    assert forbidden.isdisjoint(contract["core_routing_features"])
