from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest

from scripts.v5_dataset import REGISTRATION_FIELDS, load_v5_rows, safe_registration_view


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET = PROJECT_ROOT / "data" / "raw" / "Обращения_1931.xlsx"

requires_private_dataset = pytest.mark.skipif(
    not DATASET.exists(),
    reason="The original Service Desk XLSX is intentionally not published.",
)


@requires_private_dataset
def test_v5_dataset_is_rebuilt_from_original_xlsx_with_expected_targets() -> None:
    rows = load_v5_rows(DATASET)

    assert len(rows) == 1931
    assert len({row["request_id"] for row in rows}) == 1931
    assert len({row["category"] for row in rows}) == 43
    assert Counter(row["routing_target"] for row in rows) == {
        "(1 линия)": 1192,
        "(2 линия)": 566,
        "(3 линия)": 172,
        "(4 линия)": 1,
    }


@requires_private_dataset
def test_safe_registration_view_exposes_only_registration_time_features() -> None:
    row = load_v5_rows(DATASET)[0]
    view = safe_registration_view(row)

    assert tuple(view) == REGISTRATION_FIELDS
    assert "request_id" not in view
    assert "category" not in view
    assert "routing_target" not in view


@requires_private_dataset
def test_missing_registration_values_remain_none_not_meaningful_text() -> None:
    rows = load_v5_rows(DATASET)
    missing_component = next(row for row in rows if row["component"] is None)

    assert missing_component["component"] is None
    assert safe_registration_view(missing_component)["component"] is None


@requires_private_dataset
def test_registration_date_is_normalized_but_not_used_as_request_identity() -> None:
    rows = load_v5_rows(DATASET)

    assert all("T" in row["registration_date"] for row in rows)
    assert all(row["registration_date"] != row["request_id"] for row in rows)
