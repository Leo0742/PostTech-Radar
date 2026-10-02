from __future__ import annotations

from copy import deepcopy

import numpy as np
from app.ml.v3.metrics import paired_bootstrap_accuracy, risk_coverage_curve
from app.ml.v3.protocol import audit_manifest, build_v3_manifest


def _rows() -> list[dict[str, str]]:
    rows = []
    for label in ("A", "B", "C"):
        for group in range(12):
            for duplicate in range(2):
                rows.append(
                    {
                        "request_id": f"{label}-{group}-{duplicate}",
                        "category": label,
                        "description": f"Описание {label} {group}",
                        "normalized_description": f"описание {label.lower()} {group}",
                    }
                )
    return rows


def test_v3_manifest_is_deterministic_hashed_and_group_disjoint() -> None:
    first = build_v3_manifest(_rows(), seed=20260916, repeats=2, n_splits=3)
    second = build_v3_manifest(_rows(), seed=20260916, repeats=2, n_splits=3)

    assert first.to_dict() == second.to_dict()
    assert len(first.dataset_sha256) == 64
    assert len(first.split_sha256) == 64
    assert len(first.folds) == 6
    assert audit_manifest(first) == {
        "development_calibration_overlap": 0,
        "development_holdout_overlap": 0,
        "calibration_holdout_overlap": 0,
        "fold_group_overlaps": [0, 0, 0, 0, 0, 0],
        "all_disjoint": True,
    }


def test_dataset_hash_changes_when_a_target_changes_but_request_ids_do_not() -> None:
    original = _rows()
    changed = deepcopy(original)
    changed[0]["category"] = "B"

    assert build_v3_manifest(original).dataset_sha256 != build_v3_manifest(changed).dataset_sha256


def test_risk_coverage_and_paired_bootstrap_use_identical_examples() -> None:
    labels = ["A", "B"]
    truth = ["A", "B", "A", "B"]
    probabilities = np.asarray([[0.99, 0.01], [0.80, 0.20], [0.60, 0.40], [0.49, 0.51]])

    curve = risk_coverage_curve(truth, probabilities, labels, coverages=(0.5, 1.0))
    comparison = paired_bootstrap_accuracy(truth, ["A", "B", "A", "B"], ["A", "A", "A", "A"], seed=7, samples=500)

    assert curve == [
        {"target_coverage": 0.5, "coverage": 0.5, "accepted": 2, "errors": 1, "accepted_accuracy": 0.5},
        {"target_coverage": 1.0, "coverage": 1.0, "accepted": 4, "errors": 1, "accepted_accuracy": 0.75},
    ]
    assert comparison["accuracy_a"] == 1.0
    assert comparison["accuracy_b"] == 0.5
    assert comparison["mean_difference"] == 0.5
    assert comparison["paired_examples"] == 4
