from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold


def _sha256(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _group_value(row: dict[str, Any]) -> str:
    normalized = " ".join(str(row.get("normalized_description") or row.get("description") or "").lower().split())
    return normalized or f"request:{row['request_id']}"


def _group_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:20]


def _safe_n_splits(labels: list[str], groups: list[str], requested: int) -> int:
    grouped_labels: dict[str, set[str]] = {}
    for label, group in zip(labels, groups, strict=True):
        grouped_labels.setdefault(label, set()).add(group)
    minimum = min((len(values) for values in grouped_labels.values()), default=0)
    if minimum < 2:
        raise ValueError("Каждый класс должен иметь минимум две независимые description-группы")
    return max(2, min(requested, minimum))


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
class EvaluationManifest:
    protocol_version: str
    seed: int
    dataset_sha256: str
    split_sha256: str
    labels: tuple[str, ...]
    row_count: int
    development_request_ids: tuple[str, ...]
    calibration_request_ids: tuple[str, ...]
    holdout_request_ids: tuple[str, ...]
    development_group_hashes: tuple[str, ...]
    calibration_group_hashes: tuple[str, ...]
    holdout_group_hashes: tuple[str, ...]
    folds: tuple[FoldManifest, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def manifest_from_dict(payload: dict[str, Any]) -> EvaluationManifest:
    return EvaluationManifest(
        protocol_version=str(payload["protocol_version"]),
        seed=int(payload["seed"]),
        dataset_sha256=str(payload["dataset_sha256"]),
        split_sha256=str(payload["split_sha256"]),
        labels=tuple(payload["labels"]),
        row_count=int(payload["row_count"]),
        development_request_ids=tuple(payload["development_request_ids"]),
        calibration_request_ids=tuple(payload["calibration_request_ids"]),
        holdout_request_ids=tuple(payload["holdout_request_ids"]),
        development_group_hashes=tuple(payload["development_group_hashes"]),
        calibration_group_hashes=tuple(payload["calibration_group_hashes"]),
        holdout_group_hashes=tuple(payload["holdout_group_hashes"]),
        folds=tuple(FoldManifest(**{**item, "train_request_ids": tuple(item["train_request_ids"]),
                                    "validation_request_ids": tuple(item["validation_request_ids"]),
                                    "train_group_hashes": tuple(item["train_group_hashes"]),
                                    "validation_group_hashes": tuple(item["validation_group_hashes"])})
                    for item in payload["folds"]),
    )


def _first_split(indices: np.ndarray, labels: list[str], groups: list[str], seed: int, requested: int = 5) -> tuple[np.ndarray, np.ndarray]:
    splits = _safe_n_splits(labels, groups, requested)
    splitter = StratifiedGroupKFold(n_splits=splits, shuffle=True, random_state=seed)
    train, held = next(splitter.split(indices, labels, groups))
    return indices[train], indices[held]


def build_v3_manifest(
    rows: list[dict[str, Any]], *, seed: int = 20260916, repeats: int = 2, n_splits: int = 4
) -> EvaluationManifest:
    if repeats < 1:
        raise ValueError("repeats must be at least 1")
    ordered = sorted(rows, key=lambda row: str(row["request_id"]))
    request_ids = [str(row["request_id"]) for row in ordered]
    if len(request_ids) != len(set(request_ids)):
        raise ValueError("request_id must be unique")
    labels = [str(row["category"]) for row in ordered]
    group_values = [_group_value(row) for row in ordered]
    group_hashes = [_group_hash(value) for value in group_values]
    indices = np.arange(len(ordered))
    pool, holdout = _first_split(indices, labels, group_hashes, seed)
    pool_labels = [labels[index] for index in pool]
    pool_groups = [group_hashes[index] for index in pool]
    development, calibration = _first_split(pool, pool_labels, pool_groups, seed + 1)

    fold_manifests: list[FoldManifest] = []
    dev_labels = [labels[index] for index in development]
    dev_groups = [group_hashes[index] for index in development]
    fold_count = _safe_n_splits(dev_labels, dev_groups, n_splits)
    for repeat in range(repeats):
        repeat_seed = seed + 100 + repeat
        splitter = StratifiedGroupKFold(n_splits=fold_count, shuffle=True, random_state=repeat_seed)
        for fold, (train_relative, validation_relative) in enumerate(
            splitter.split(np.arange(len(development)), dev_labels, dev_groups)
        ):
            train = development[train_relative]
            validation = development[validation_relative]
            fold_manifests.append(
                FoldManifest(
                    repeat=repeat,
                    fold=fold,
                    seed=repeat_seed,
                    train_request_ids=tuple(request_ids[index] for index in train),
                    validation_request_ids=tuple(request_ids[index] for index in validation),
                    train_group_hashes=tuple(sorted({group_hashes[index] for index in train})),
                    validation_group_hashes=tuple(sorted({group_hashes[index] for index in validation})),
                )
            )

    canonical_rows = [
        {
            "request_id": request_ids[index],
            "category": labels[index],
            "group_hash": group_hashes[index],
            "row": ordered[index],
        }
        for index in indices
    ]
    dataset_sha256 = _sha256(canonical_rows)
    split_payload = {
        "protocol_version": "3.0",
        "seed": seed,
        "dataset_sha256": dataset_sha256,
        "development_request_ids": [request_ids[index] for index in development],
        "calibration_request_ids": [request_ids[index] for index in calibration],
        "holdout_request_ids": [request_ids[index] for index in holdout],
        "folds": [asdict(item) for item in fold_manifests],
    }
    return EvaluationManifest(
        protocol_version="3.0",
        seed=seed,
        dataset_sha256=dataset_sha256,
        split_sha256=_sha256(split_payload),
        labels=tuple(sorted(Counter(labels))),
        row_count=len(ordered),
        development_request_ids=tuple(request_ids[index] for index in development),
        calibration_request_ids=tuple(request_ids[index] for index in calibration),
        holdout_request_ids=tuple(request_ids[index] for index in holdout),
        development_group_hashes=tuple(sorted({group_hashes[index] for index in development})),
        calibration_group_hashes=tuple(sorted({group_hashes[index] for index in calibration})),
        holdout_group_hashes=tuple(sorted({group_hashes[index] for index in holdout})),
        folds=tuple(fold_manifests),
    )


def audit_manifest(manifest: EvaluationManifest) -> dict[str, Any]:
    development = set(manifest.development_group_hashes)
    calibration = set(manifest.calibration_group_hashes)
    holdout = set(manifest.holdout_group_hashes)
    fold_overlaps = [
        len(set(fold.train_group_hashes) & set(fold.validation_group_hashes))
        for fold in manifest.folds
    ]
    result = {
        "development_calibration_overlap": len(development & calibration),
        "development_holdout_overlap": len(development & holdout),
        "calibration_holdout_overlap": len(calibration & holdout),
        "fold_group_overlaps": fold_overlaps,
    }
    result["all_disjoint"] = all(value == 0 for key, value in result.items() if key != "fold_group_overlaps") and all(
        value == 0 for value in fold_overlaps
    )
    return result
