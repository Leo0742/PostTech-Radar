from __future__ import annotations

from pathlib import Path
import subprocess
import sys

from scripts.v5_protocol import (
    audit_protocol,
    build_v5_protocol,
    group_hash,
    load_protocol_rows,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET = PROJECT_ROOT / "data" / "raw" / "Обращения_1931.xlsx"


def test_protocol_script_can_be_invoked_directly() -> None:
    result = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "scripts" / "v5_protocol.py"), "--help"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def test_group_hash_collapses_case_and_whitespace_duplicates() -> None:
    assert group_hash("  QR  Код\nне РАБОТАЕТ ") == group_hash("qr код не работает")


def test_protocol_rows_are_rebuilt_from_original_xlsx() -> None:
    rows = load_protocol_rows(DATASET)

    assert len(rows) == 1931
    assert len({row["request_id"] for row in rows}) == 1931
    assert all(row["category"] for row in rows)
    assert all(row["registration_date"] for row in rows)


def test_v5_protocol_is_deterministic_and_has_repeated_grouped_folds() -> None:
    rows = load_protocol_rows(DATASET)
    first = build_v5_protocol(rows)
    second = build_v5_protocol(rows)

    assert first["split_sha256"] == second["split_sha256"]
    assert first["dataset_sha256"] == second["dataset_sha256"]
    assert first["protocol_version"] == "5.1"
    assert first["internal_lockbox_name"] == "V5 INTERNAL LOCKBOX"
    assert len(first["folds"]) == 12
    assert {fold["repeat"] for fold in first["folds"]} == {0, 1, 2}
    assert {fold["fold"] for fold in first["folds"]} == {0, 1, 2, 3}


def test_v5_protocol_has_zero_group_overlap_everywhere() -> None:
    protocol = build_v5_protocol(load_protocol_rows(DATASET))
    audit = audit_protocol(protocol)

    assert audit["all_group_overlaps_zero"] is True
    assert audit["synthetic_evaluation_rows"] == 0
    assert all(value == 0 for value in audit["partition_group_overlaps"].values())
    assert all(value == 0 for value in audit["fold_group_overlaps"])
