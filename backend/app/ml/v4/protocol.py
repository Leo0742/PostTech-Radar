from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold


def _sha256(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _group_value(row: dict[str, Any]) -> str:
    value = " ".join(str(row.get("normalized_description") or row.get("description") or "").lower().split())
    return value or f"request:{row['request_id']}"


def _group_hash(row: dict[str, Any]) -> str:
    return hashlib.sha256(_group_value(row).encode("utf-8")).hexdigest()[:20]


def _timestamp(row: dict[str, Any]) -> str:
    value = str(row.get("registration_date") or "")
    if not value:
        raise ValueError("registration_date is required for the v4 temporal protocol")
    datetime.fromisoformat(value.replace("Z", "+00:00"))
    return value


@dataclass(frozen=True)
class FoldManifest:
    repeat: int
    fold: int
    seed: int
    train_request_ids: tuple[str, ...]
    validation_request_ids: tuple[str, ...]
    train_group_hashes: tuple[str, ...]
    validation_group_hashes: tuple[str, ...]


@dataclass(frozen=True)
class V4Protocol:
    protocol_version: str
    seed: int
    dataset_sha256: str
    split_sha256: str
    real_row_count: int
    labels: tuple[str, ...]
    development_request_ids: tuple[str, ...]
    calibration_request_ids: tuple[str, ...]
    holdout_request_ids: tuple[str, ...]
    development_group_hashes: tuple[str, ...]
    calibration_group_hashes: tuple[str, ...]
    holdout_group_hashes: tuple[str, ...]
    folds: tuple[FoldManifest, ...]
    temporal_train_request_ids: tuple[str, ...]
    temporal_test_request_ids: tuple[str, ...]
    temporal_train_group_hashes: tuple[str, ...]
    temporal_test_group_hashes: tuple[str, ...]
    temporal_train_timestamps: tuple[str, ...]
    temporal_test_timestamps: tuple[str, ...]
    retrieval_development_request_ids: tuple[str, ...]
    retrieval_test_request_ids: tuple[str, ...]
    retrieval_development_group_hashes: tuple[str, ...]
    retrieval_test_group_hashes: tuple[str, ...]
    ood_categories: tuple[str, ...]
    ood_known_request_ids: tuple[str, ...]
    ood_request_ids: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _common_partition(
    indices: np.ndarray,
    labels: list[str],
    groups: list[str],
    *,
    seed: int,
    parts: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    splitter = StratifiedGroupKFold(n_splits=parts, shuffle=True, random_state=seed)
    keep_relative, held_relative = next(splitter.split(indices, [labels[index] for index in indices], [groups[index] for index in indices]))
    return indices[keep_relative], indices[held_relative]


def _group_counts(labels: list[str], groups: list[str]) -> dict[str, int]:
    values: dict[str, set[str]] = defaultdict(set)
    for label, group in zip(labels, groups, strict=True):
        values[label].add(group)
    return {label: len(items) for label, items in values.items()}


def _development_folds(
    development: np.ndarray,
    labels: list[str],
    groups: list[str],
    request_ids: list[str],
    *,
    seed: int,
    repeats: int,
    n_splits: int,
) -> list[FoldManifest]:
    counts = _group_counts([labels[index] for index in development], [groups[index] for index in development])
    common = np.asarray([index for index in development if counts[labels[index]] >= n_splits], dtype=int)
    rare_groups = sorted({groups[index] for index in development if counts[labels[index]] < n_splits})
    result: list[FoldManifest] = []
    for repeat in range(repeats):
        repeat_seed = seed + 100 + repeat
        common_validation: dict[int, set[str]] = {fold: set() for fold in range(n_splits)}
        if len(common):
            splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=repeat_seed)
            for fold, (_train, validation) in enumerate(
                splitter.split(common, [labels[index] for index in common], [groups[index] for index in common])
            ):
                common_validation[fold] = {groups[int(common[position])] for position in validation}
        rare_validation: dict[int, set[str]] = {fold: set() for fold in range(n_splits)}
        ordered_rare = sorted(rare_groups, key=lambda value: _sha256([repeat_seed, value]))
        for position, group in enumerate(ordered_rare):
            rare_validation[position % n_splits].add(group)
        development_groups = {groups[index] for index in development}
        for fold in range(n_splits):
            validation_groups = common_validation[fold] | rare_validation[fold]
            train_groups = development_groups - validation_groups
            result.append(
                FoldManifest(
                    repeat=repeat,
                    fold=fold,
                    seed=repeat_seed,
                    train_request_ids=tuple(request_ids[index] for index in development if groups[index] in train_groups),
                    validation_request_ids=tuple(
                        request_ids[index] for index in development if groups[index] in validation_groups
                    ),
                    train_group_hashes=tuple(sorted(train_groups)),
                    validation_group_hashes=tuple(sorted(validation_groups)),
                )
            )
    return result


def build_v4_protocol(
    rows: list[dict[str, Any]],
    *,
    seed: int = 20260917,
    repeats: int = 3,
    n_splits: int = 4,
    ood_category_count: int = 5,
) -> V4Protocol:
    if repeats < 1 or n_splits < 2:
        raise ValueError("repeats must be >=1 and n_splits must be >=2")
    real_rows = [row for row in rows if not bool(row.get("is_synthetic", False))]
    ordered = sorted(real_rows, key=lambda row: str(row["request_id"]))
    request_ids = [str(row["request_id"]) for row in ordered]
    if len(request_ids) != len(set(request_ids)):
        raise ValueError("request_id must be unique")
    if not ordered:
        raise ValueError("at least one real row is required")
    labels = [str(row["category"]) for row in ordered]
    groups = [_group_hash(row) for row in ordered]
    timestamps = [_timestamp(row) for row in ordered]
    indices = np.arange(len(ordered), dtype=int)

    counts = _group_counts(labels, groups)
    eligible = np.asarray([index for index in indices if counts[labels[index]] >= 5], dtype=int)
    rare = np.asarray([index for index in indices if counts[labels[index]] < 5], dtype=int)
    if len({labels[index] for index in eligible}) < 2:
        raise ValueError("at least two categories with five independent groups are required")
    pool, holdout = _common_partition(eligible, labels, groups, seed=seed)
    common_development, calibration = _common_partition(pool, labels, groups, seed=seed + 1)
    development = np.asarray(sorted([*map(int, common_development), *map(int, rare)]), dtype=int)
    folds = _development_folds(
        development,
        labels,
        groups,
        request_ids,
        seed=seed,
        repeats=repeats,
        n_splits=n_splits,
    )

    group_timestamps: dict[str, str] = {}
    for group, timestamp in zip(groups, timestamps, strict=True):
        group_timestamps[group] = max(group_timestamps.get(group, timestamp), timestamp)
    ordered_groups = sorted(group_timestamps, key=lambda group: (group_timestamps[group], group))
    temporal_test_count = max(1, int(round(len(ordered_groups) * 0.2)))
    temporal_test_groups = set(ordered_groups[-temporal_test_count:])
    temporal_train_groups = set(ordered_groups[:-temporal_test_count])

    retrieval_order = sorted(set(groups), key=lambda group: _sha256([seed, "retrieval", group]))
    retrieval_test_count = max(1, int(round(len(retrieval_order) * 0.2)))
    retrieval_test_groups = set(retrieval_order[:retrieval_test_count])
    retrieval_development_groups = set(retrieval_order[retrieval_test_count:])

    category_counts = _group_counts(labels, groups)
    ood_candidates = sorted(category_counts, key=lambda label: (_sha256([seed, "ood", label]), label))
    selected_ood = tuple(ood_candidates[: min(max(1, ood_category_count), len(ood_candidates) - 1)])
    selected_ood_set = set(selected_ood)

    canonical_rows = [
        {
            "request_id": request_ids[index],
            "category": labels[index],
            "group_hash": groups[index],
            "registration_date": timestamps[index],
            "row": ordered[index],
        }
        for index in indices
    ]
    dataset_sha256 = _sha256(canonical_rows)
    split_payload = {
        "protocol_version": "4.0",
        "seed": seed,
        "dataset_sha256": dataset_sha256,
        "development": [request_ids[index] for index in development],
        "calibration": [request_ids[index] for index in calibration],
        "holdout": [request_ids[index] for index in holdout],
        "folds": [asdict(item) for item in folds],
        "temporal_test_groups": sorted(temporal_test_groups),
        "retrieval_test_groups": sorted(retrieval_test_groups),
        "ood_categories": selected_ood,
    }
    split_sha256 = _sha256(split_payload)
    return V4Protocol(
        protocol_version="4.0",
        seed=seed,
        dataset_sha256=dataset_sha256,
        split_sha256=split_sha256,
        real_row_count=len(ordered),
        labels=tuple(sorted(set(labels))),
        development_request_ids=tuple(request_ids[index] for index in development),
        calibration_request_ids=tuple(request_ids[index] for index in calibration),
        holdout_request_ids=tuple(request_ids[index] for index in holdout),
        development_group_hashes=tuple(sorted({groups[index] for index in development})),
        calibration_group_hashes=tuple(sorted({groups[index] for index in calibration})),
        holdout_group_hashes=tuple(sorted({groups[index] for index in holdout})),
        folds=tuple(folds),
        temporal_train_request_ids=tuple(request_ids[index] for index in indices if groups[index] in temporal_train_groups),
        temporal_test_request_ids=tuple(request_ids[index] for index in indices if groups[index] in temporal_test_groups),
        temporal_train_group_hashes=tuple(sorted(temporal_train_groups)),
        temporal_test_group_hashes=tuple(sorted(temporal_test_groups)),
        temporal_train_timestamps=tuple(group_timestamps[group] for group in sorted(temporal_train_groups, key=group_timestamps.get)),
        temporal_test_timestamps=tuple(group_timestamps[group] for group in sorted(temporal_test_groups, key=group_timestamps.get)),
        retrieval_development_request_ids=tuple(
            request_ids[index] for index in indices if groups[index] in retrieval_development_groups
        ),
        retrieval_test_request_ids=tuple(request_ids[index] for index in indices if groups[index] in retrieval_test_groups),
        retrieval_development_group_hashes=tuple(sorted(retrieval_development_groups)),
        retrieval_test_group_hashes=tuple(sorted(retrieval_test_groups)),
        ood_categories=selected_ood,
        ood_known_request_ids=tuple(request_ids[index] for index in indices if labels[index] not in selected_ood_set),
        ood_request_ids=tuple(request_ids[index] for index in indices if labels[index] in selected_ood_set),
    )


def audit_v4_protocol(protocol: V4Protocol) -> dict[str, Any]:
    overlaps = {
        "development_calibration": len(set(protocol.development_group_hashes) & set(protocol.calibration_group_hashes)),
        "development_holdout": len(set(protocol.development_group_hashes) & set(protocol.holdout_group_hashes)),
        "calibration_holdout": len(set(protocol.calibration_group_hashes) & set(protocol.holdout_group_hashes)),
        "temporal": len(set(protocol.temporal_train_group_hashes) & set(protocol.temporal_test_group_hashes)),
        "retrieval": len(
            set(protocol.retrieval_development_group_hashes) & set(protocol.retrieval_test_group_hashes)
        ),
    }
    fold_overlaps = [
        len(set(fold.train_group_hashes) & set(fold.validation_group_hashes)) for fold in protocol.folds
    ]
    return {
        "partition_group_overlaps": overlaps,
        "fold_group_overlaps": fold_overlaps,
        "all_group_overlaps_zero": all(value == 0 for value in overlaps.values())
        and all(value == 0 for value in fold_overlaps),
        "synthetic_evaluation_rows": 0,
    }


def protocol_artifact_dir(protocol: V4Protocol) -> Path:
    return Path("artifacts/gpu_research_v4/protocols") / protocol.dataset_sha256 / protocol.split_sha256
