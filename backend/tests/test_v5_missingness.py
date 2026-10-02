from __future__ import annotations

from scripts.v5_missingness import (
    OPTIONAL_FIELDS,
    SCENARIOS,
    apply_missingness_scenario,
    input_quality,
)


BASE = {
    "description": "Не работает QR-код в мобильном приложении",
    "user": "operator",
    "service": "Портал",
    "component": "Мобильное приложение",
    "request_type": "Инцидент",
    "criticality": "Средняя",
    "urgency": "Средняя",
    "priority": "(3) Средний",
    "service_class": "7x24",
    "timezone": "Москва",
}


def test_required_robustness_scenarios_are_declared() -> None:
    assert tuple(SCENARIOS) == (
        "full",
        "description_only",
        "metadata_only",
        "no_component",
        "no_service",
        "no_request_type",
        "no_priority_criticality",
        "random_dropout_10",
        "random_dropout_30",
        "random_dropout_50",
        "empirical_missingness",
    )


def test_deterministic_masks_remove_only_requested_fields() -> None:
    no_component = apply_missingness_scenario(BASE, "no_component", seed=7)
    assert no_component["component"] is None
    assert no_component["description"] == BASE["description"]

    description_only = apply_missingness_scenario(BASE, "description_only", seed=7)
    assert description_only["description"] == BASE["description"]
    assert all(description_only[field] is None for field in OPTIONAL_FIELDS if field != "description")

    metadata_only = apply_missingness_scenario(BASE, "metadata_only", seed=7)
    assert metadata_only["description"] is None
    assert metadata_only["service"] == BASE["service"]


def test_random_dropout_is_reproducible_and_does_not_mutate_input() -> None:
    first = apply_missingness_scenario(BASE, "random_dropout_30", seed=20260917)
    second = apply_missingness_scenario(BASE, "random_dropout_30", seed=20260917)

    assert first == second
    assert BASE["component"] == "Мобильное приложение"


def test_input_quality_is_separate_from_model_confidence() -> None:
    full = input_quality(BASE)
    sparse = input_quality(apply_missingness_scenario(BASE, "metadata_only", seed=1))
    empty = input_quality({key: None for key in BASE})

    assert full["input_completeness"] == 1.0
    assert sparse["input_completeness"] < full["input_completeness"]
    assert empty["informative_field_count"] == 0
    assert empty["low_information_warning"] is True
    assert "model_confidence" not in full
