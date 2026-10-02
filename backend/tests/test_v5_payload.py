from __future__ import annotations

import hashlib
import json
import tarfile
from pathlib import Path

from scripts.v5.build_server_payload import build_payload

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_payload_contains_v5_execution_inputs_and_excludes_bulk_v4_runtime(tmp_path: Path) -> None:
    archive = tmp_path / "payload.tar.gz"

    result = build_payload(PROJECT_ROOT, archive)

    assert result["files"] > 10
    with tarfile.open(archive, "r:gz") as tar:
        names = set(tar.getnames())
    required = {
        "posttech-v5.1/data/raw/Обращения_1931.xlsx",
        "posttech-v5.1/artifacts/gpu_research_v5/data_contract.json",
        "posttech-v5.1/artifacts/gpu_research_v5/protocol/protocol.json",
        "posttech-v5.1/configs/v5/server_a_qwen8b.json",
        "posttech-v5.1/configs/v5/server_b_challengers.json",
        "posttech-v5.1/scripts/v5_gpu_runner.py",
        "posttech-v5.1/scripts/v5/server_bootstrap.sh",
        "posttech-v5.1/requirements-v5-gpu.txt",
        "posttech-v5.1/SHA256SUMS.json",
    }
    assert required.issubset(names)
    assert not any(".venv" in name for name in names)
    assert not any("artifacts/gpu_research_v5/embedding_cache" in name for name in names)
    assert not any(name.startswith("posttech-v5.1/models/") for name in names)
    assert not any(name.startswith("posttech-v5.1/frontend/") for name in names)


def test_payload_manifest_hashes_match_archived_files(tmp_path: Path) -> None:
    archive = tmp_path / "payload.tar.gz"
    build_payload(PROJECT_ROOT, archive)
    extract = tmp_path / "extract"
    with tarfile.open(archive, "r:gz") as tar:
        tar.extractall(extract, filter="data")
    root = extract / "posttech-v5.1"
    manifest = json.loads((root / "SHA256SUMS.json").read_text())

    for relative, expected in manifest["files"].items():
        assert _sha256(root / relative) == expected


def test_payload_build_is_deterministic_for_unchanged_sources(tmp_path: Path) -> None:
    first = tmp_path / "first.tar.gz"
    second = tmp_path / "second.tar.gz"

    build_payload(PROJECT_ROOT, first)
    build_payload(PROJECT_ROOT, second)

    assert _sha256(first) == _sha256(second)
