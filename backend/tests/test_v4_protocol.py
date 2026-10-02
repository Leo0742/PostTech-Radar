from __future__ import annotations

from datetime import datetime, timedelta

import pytest
from app.ml.v4.protocol import audit_v4_protocol, build_v4_protocol, protocol_artifact_dir


def _rows() -> list[dict[str, object]]:
    start = datetime(2025, 1, 1)
    rows: list[dict[str, object]] = []
    request = 0
    for label, groups in (("common-a", 8), ("common-b", 8), ("rare", 2), ("singleton", 1)):
        for group in range(groups):
            request += 1
            rows.append(
                {
                    "request_id": str(request),
                    "category": label,
                    "description": f"{label} text {group}",
                    "normalized_description": f"{label} group {group}",
                    "registration_date": (start + timedelta(days=request)).isoformat(),
                    "is_synthetic": False,
                }
            )
    return rows


def test_v4_protocol_keeps_synthetic_out_of_every_evaluation_partition() -> None:
    rows = _rows()
    rows.append(
        {
            "request_id": "synthetic-1",
            "category": "rare",
            "description": "generated",
            "normalized_description": "generated",
            "registration_date": "2025-12-31T00:00:00",
            "is_synthetic": True,
        }
    )

    protocol = build_v4_protocol(rows, repeats=2, n_splits=4)

    serialized = protocol.to_dict()
    assert "synthetic-1" not in repr(serialized)
    assert protocol.real_row_count == len(rows) - 1
    assert set(protocol.labels) == {"common-a", "common-b", "rare", "singleton"}


def test_v4_protocol_has_zero_group_overlap_and_full_oof_coverage() -> None:
    protocol = build_v4_protocol(_rows(), repeats=2, n_splits=4)

    audit = audit_v4_protocol(protocol)

    assert audit["all_group_overlaps_zero"] is True
    assert audit["synthetic_evaluation_rows"] == 0
    for repeat in range(2):
        validation_ids = {
            request_id
            for fold in protocol.folds
            if fold.repeat == repeat
            for request_id in fold.validation_request_ids
        }
        assert validation_ids == set(protocol.development_request_ids)


def test_temporal_and_retrieval_tests_are_independent_and_group_disjoint() -> None:
    protocol = build_v4_protocol(_rows(), repeats=1, n_splits=4)

    assert set(protocol.temporal_train_group_hashes).isdisjoint(protocol.temporal_test_group_hashes)
    assert set(protocol.retrieval_development_group_hashes).isdisjoint(protocol.retrieval_test_group_hashes)
    assert max(protocol.temporal_train_timestamps) < min(protocol.temporal_test_timestamps)


def test_leave_category_out_removes_complete_categories_from_known_training() -> None:
    protocol = build_v4_protocol(_rows(), repeats=1, n_splits=4, ood_category_count=2)

    assert len(protocol.ood_categories) == 2
    assert set(protocol.ood_request_ids).isdisjoint(protocol.ood_known_request_ids)
    labels_by_id = {str(row["request_id"]): str(row["category"]) for row in _rows()}
    assert {labels_by_id[item] for item in protocol.ood_request_ids} == set(protocol.ood_categories)
    assert not ({labels_by_id[item] for item in protocol.ood_known_request_ids} & set(protocol.ood_categories))


def test_protocol_artifacts_are_versioned_by_dataset_and_split_hash() -> None:
    protocol = build_v4_protocol(_rows(), repeats=1, n_splits=4)

    path = protocol_artifact_dir(protocol)

    assert path.as_posix() == f"artifacts/gpu_research_v4/protocols/{protocol.dataset_sha256}/{protocol.split_sha256}"


def test_duplicate_request_ids_are_rejected() -> None:
    rows = _rows()
    rows.append(dict(rows[0]))

    with pytest.raises(ValueError, match="request_id"):
        build_v4_protocol(rows)


def test_future_cumulative_rows_create_a_new_versioned_protocol() -> None:
    rows = _rows()
    first = build_v4_protocol(rows, repeats=1, n_splits=4)
    rows.append(
        {
            "request_id": "future-10000",
            "category": "future-category",
            "description": "new accumulated labeled ticket",
            "normalized_description": "new accumulated labeled ticket",
            "registration_date": "2026-01-01T00:00:00",
            "is_synthetic": False,
        }
    )

    updated = build_v4_protocol(rows, repeats=1, n_splits=4)

    assert updated.real_row_count == first.real_row_count + 1
    assert updated.dataset_sha256 != first.dataset_sha256
    assert "future-category" in updated.labels
