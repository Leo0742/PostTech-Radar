from __future__ import annotations

import argparse
import hashlib
import json
import sys
import unicodedata
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.v5_data_audit import read_xlsx_rows


DEFAULT_DATASET = PROJECT_ROOT / "data" / "raw" / "Обращения_1931.xlsx"
PROTOCOL_ROOT = PROJECT_ROOT / "artifacts" / "gpu_research_v5" / "protocol"
OUTPUT_ROOT = PROJECT_ROOT / "outputs"


def _sha256_json(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_description(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).lower()
    return " ".join(text.split())


def group_hash(value: Any, *, request_id: str | None = None) -> str:
    normalized = normalize_description(value)
    key = normalized or (f"request:{request_id}" if request_id is not None else "")
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:20]


def _request_id(value: Any) -> str:
    if value in {None, ""}:
        raise ValueError("Номер запроса cannot be empty")
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value).strip()


def _excel_datetime(value: Any) -> str:
    if value in {None, ""}:
        raise ValueError("Дата регистрации cannot be empty")
    if isinstance(value, (int, float)):
        moment = datetime(1899, 12, 30) + timedelta(days=float(value))
        return moment.isoformat(timespec="microseconds")
    text = str(value).strip()
    datetime.fromisoformat(text.replace("Z", "+00:00"))
    return text


def load_protocol_rows(path: Path) -> list[dict[str, Any]]:
    _sheet, _headers, source_rows, _storage = read_xlsx_rows(Path(path))
    rows: list[dict[str, Any]] = []
    for source in source_rows:
        request_id = _request_id(source.get("Номер запроса"))
        category = str(source.get("Вид запроса") or "").strip()
        if not category:
            raise ValueError(f"Request {request_id} has empty category")
        rows.append({"request_id": request_id, "category": category, "description": str(source.get("Описание 2") or ""), "registration_date": _excel_datetime(source.get("Дата регистрации"))})
    return rows


def _group_counts(labels: list[str], groups: list[str]) -> dict[str, int]:
    values: dict[str, set[str]] = defaultdict(set)
    for label, group in zip(labels, groups, strict=True):
        values[label].add(group)
    return {label: len(items) for label, items in values.items()}


def _common_partition(indices: np.ndarray, labels: list[str], groups: list[str], *, seed: int, parts: int = 5) -> tuple[np.ndarray, np.ndarray]:
    splitter = StratifiedGroupKFold(n_splits=parts, shuffle=True, random_state=seed)
    keep_relative, held_relative = next(splitter.split(indices, [labels[index] for index in indices], [groups[index] for index in indices]))
    return indices[keep_relative], indices[held_relative]


def _development_folds(development: np.ndarray, labels: list[str], groups: list[str], request_ids: list[str], *, seeds: tuple[int, ...], n_splits: int) -> list[dict[str, Any]]:
    counts = _group_counts([labels[index] for index in development], [groups[index] for index in development])
    common = np.asarray([index for index in development if counts[labels[index]] >= n_splits], dtype=int)
    rare_groups = sorted({groups[index] for index in development if counts[labels[index]] < n_splits})
    development_groups = {groups[index] for index in development}
    result: list[dict[str, Any]] = []
    for repeat, repeat_seed in enumerate(seeds):
        common_validation: dict[int, set[str]] = {fold: set() for fold in range(n_splits)}
        if len(common):
            splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=repeat_seed)
            for fold, (_train, validation) in enumerate(splitter.split(common, [labels[index] for index in common], [groups[index] for index in common])):
                common_validation[fold] = {groups[int(common[position])] for position in validation}
        rare_validation: dict[int, set[str]] = {fold: set() for fold in range(n_splits)}
        ordered_rare = sorted(rare_groups, key=lambda value: _sha256_json([repeat_seed, value]))
        for position, group in enumerate(ordered_rare):
            rare_validation[position % n_splits].add(group)
        for fold in range(n_splits):
            validation_groups = common_validation[fold] | rare_validation[fold]
            train_groups = development_groups - validation_groups
            result.append({
                "repeat": repeat,
                "fold": fold,
                "seed": repeat_seed,
                "train_request_ids": [request_ids[index] for index in development if groups[index] in train_groups],
                "validation_request_ids": [request_ids[index] for index in development if groups[index] in validation_groups],
                "train_group_hashes": sorted(train_groups),
                "validation_group_hashes": sorted(validation_groups),
            })
    return result


def build_v5_protocol(rows: list[dict[str, Any]], *, seed: int = 20260917, repeat_seeds: tuple[int, ...] = (20260917, 20260918, 20260919), n_splits: int = 4) -> dict[str, Any]:
    if n_splits < 2 or not repeat_seeds:
        raise ValueError("n_splits must be >= 2 and repeat_seeds must be non-empty")
    ordered = sorted(rows, key=lambda row: str(row["request_id"]))
    if not ordered:
        raise ValueError("At least one real row is required")
    request_ids = [str(row["request_id"]) for row in ordered]
    if len(request_ids) != len(set(request_ids)):
        raise ValueError("request_id must be unique")
    labels = [str(row["category"]) for row in ordered]
    groups = [group_hash(row.get("description"), request_id=str(row["request_id"])) for row in ordered]
    timestamps = [str(row["registration_date"]) for row in ordered]
    indices = np.arange(len(ordered), dtype=int)
    group_counts = _group_counts(labels, groups)
    eligible = np.asarray([index for index in indices if group_counts[labels[index]] >= 5], dtype=int)
    rare = np.asarray([index for index in indices if group_counts[labels[index]] < 5], dtype=int)
    if len({labels[index] for index in eligible}) < 2:
        raise ValueError("Need at least two categories with five independent groups")
    pool, internal_lockbox = _common_partition(eligible, labels, groups, seed=seed)
    common_development, calibration = _common_partition(pool, labels, groups, seed=seed + 1)
    development = np.asarray(sorted([*map(int, common_development), *map(int, rare)]), dtype=int)
    folds = _development_folds(development, labels, groups, request_ids, seeds=repeat_seeds, n_splits=n_splits)
    group_timestamps: dict[str, str] = {}
    for group, timestamp in zip(groups, timestamps, strict=True):
        group_timestamps[group] = max(group_timestamps.get(group, timestamp), timestamp)
    ordered_groups = sorted(group_timestamps, key=lambda group: (group_timestamps[group], group))
    temporal_test_count = max(1, int(round(len(ordered_groups) * 0.20)))
    temporal_test_groups = set(ordered_groups[-temporal_test_count:])
    temporal_train_groups = set(ordered_groups[:-temporal_test_count])
    canonical_rows = [{"request_id": request_ids[index], "category": labels[index], "group_hash": groups[index], "registration_date": timestamps[index]} for index in indices]
    dataset_sha256 = _sha256_json(canonical_rows)
    protocol: dict[str, Any] = {
        "protocol_version": "5.1",
        "internal_lockbox_name": "V5 INTERNAL LOCKBOX",
        "seed": seed,
        "repeat_seeds": list(repeat_seeds),
        "n_splits": n_splits,
        "dataset_sha256": dataset_sha256,
        "real_row_count": len(ordered),
        "labels": sorted(set(labels)),
        "development_request_ids": [request_ids[index] for index in development],
        "calibration_request_ids": [request_ids[index] for index in calibration],
        "internal_lockbox_request_ids": [request_ids[index] for index in internal_lockbox],
        "development_group_hashes": sorted({groups[index] for index in development}),
        "calibration_group_hashes": sorted({groups[index] for index in calibration}),
        "internal_lockbox_group_hashes": sorted({groups[index] for index in internal_lockbox}),
        "folds": folds,
        "temporal_train_request_ids": [request_ids[index] for index in indices if groups[index] in temporal_train_groups],
        "temporal_test_request_ids": [request_ids[index] for index in indices if groups[index] in temporal_test_groups],
        "temporal_train_group_hashes": sorted(temporal_train_groups),
        "temporal_test_group_hashes": sorted(temporal_test_groups),
        "rare_category_policy": "categories with <5 independent groups remain development-only for grouped tuning; report them separately and do not pretend lockbox reliability",
        "synthetic_policy": "train_only",
    }
    split_payload = {key: protocol[key] for key in ("protocol_version", "seed", "repeat_seeds", "n_splits", "dataset_sha256", "development_request_ids", "calibration_request_ids", "internal_lockbox_request_ids", "folds", "temporal_test_group_hashes")}
    protocol["split_sha256"] = _sha256_json(split_payload)
    return protocol


def audit_protocol(protocol: dict[str, Any]) -> dict[str, Any]:
    development = set(protocol["development_group_hashes"])
    calibration = set(protocol["calibration_group_hashes"])
    lockbox = set(protocol["internal_lockbox_group_hashes"])
    overlaps = {
        "development_calibration": len(development & calibration),
        "development_internal_lockbox": len(development & lockbox),
        "calibration_internal_lockbox": len(calibration & lockbox),
        "temporal": len(set(protocol["temporal_train_group_hashes"]) & set(protocol["temporal_test_group_hashes"])),
    }
    fold_overlaps = [len(set(fold["train_group_hashes"]) & set(fold["validation_group_hashes"])) for fold in protocol["folds"]]
    return {"partition_group_overlaps": overlaps, "fold_group_overlaps": fold_overlaps, "all_group_overlaps_zero": all(value == 0 for value in overlaps.values()) and all(value == 0 for value in fold_overlaps), "synthetic_evaluation_rows": 0}


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze PostTech Radar V5.1 grouped evaluation protocol")
    parser.add_argument("dataset", nargs="?", type=Path, default=DEFAULT_DATASET)
    args = parser.parse_args()
    rows = load_protocol_rows(args.dataset)
    protocol = build_v5_protocol(rows)
    audit = audit_protocol(protocol)
    if not audit["all_group_overlaps_zero"]:
        raise RuntimeError("V5 protocol overlap audit failed")
    PROTOCOL_ROOT.mkdir(parents=True, exist_ok=True)
    (PROTOCOL_ROOT / "protocol.json").write_text(json.dumps(protocol, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"dataset_sha256": protocol["dataset_sha256"], "split_sha256": protocol["split_sha256"], "folds": len(protocol["folds"]), "overlap_audit": True}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
