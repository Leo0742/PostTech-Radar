from __future__ import annotations

import pytest
from app.core.features import (
    CATEGORY_TARGET,
    POST_PROCESSING_FIELDS,
    REGISTRATION_FEATURES,
    ROUTING_TARGET,
)
from app.services.data_service import human_duration, normalize_description, parse_duration


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("00:00:00", 0),
        ("24:00:00", 86_400),
        ("217:23:03", 782_583),
        (None, None),
        ("не время", None),
    ],
)
def test_duration_parser_supports_unbounded_hours(raw: str | None, expected: int | None) -> None:
    assert parse_duration(raw) == expected


def test_human_duration_does_not_wrap_after_24_hours() -> None:
    assert human_duration(217 * 3600 + 23 * 60 + 3) == "217 ч 23 мин"


def test_description_normalization_is_conservative() -> None:
    assert normalize_description("  QR-код\n НЕ   отображается ") == "qr-код не отображается"


def test_model_features_are_registration_only() -> None:
    forbidden = set(POST_PROCESSING_FIELDS) | {CATEGORY_TARGET, ROUTING_TARGET}
    assert forbidden.isdisjoint(REGISTRATION_FEATURES)
    assert "Описание 2" in REGISTRATION_FEATURES
