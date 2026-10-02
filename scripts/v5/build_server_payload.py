from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import tarfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "PostTech_Radar_V5_Server_Payload_2026-09-17.tar.gz"
ARCHIVE_ROOT = "posttech-v5.1"

EXACT_FILES = (
    "requirements.txt",
    "requirements-v5-gpu.txt",
    "pyproject.toml",
    "pytest.ini",
    "data/raw/Обращения_1931.xlsx",
    "data/category_definitions_v4.yaml",
    "artifacts/gpu_research_v5/data_contract.json",
    "outputs/V5_INPUT_OUTPUT_CONTRACT.md",
    "outputs/V5_EVALUATION_PROTOCOL.md",
    "outputs/BASELINES_V5.md",
    "outputs/V5_GPU_EXECUTION_HANDOFF.md",
    "scripts/v5_data_audit.py",
    "scripts/v5_dataset.py",
    "scripts/v5_protocol.py",
    "scripts/v5_missingness.py",
    "scripts/v5_metadata_baselines.py",
    "scripts/v5_experiment.py",
    "scripts/v5_gpu_runner.py",
)

GLOBS = (
    "configs/v5/*.json",
    "artifacts/gpu_research_v5/protocol/*.json",
    "artifacts/gpu_research_v5/baselines/*.json",
    "scripts/v5/*.py",
    "scripts/v5/*.sh",
    "backend/tests/test_v5*.py",
    "references/v4/*.json",
    "references/v4/*.yaml",
)


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def collect_payload_files(project_root: Path) -> list[Path]:
    root = Path(project_root)
    selected: set[Path] = set()
    for relative in EXACT_FILES:
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(f"Required payload file is missing: {relative}")
        selected.add(path)
    for pattern in GLOBS:
        selected.update(path for path in root.glob(pattern) if path.is_file())
    return sorted(selected, key=lambda path: path.relative_to(root).as_posix())


def _tar_info(relative: str, data: bytes, *, executable: bool = False) -> tarfile.TarInfo:
    info = tarfile.TarInfo(name=f"{ARCHIVE_ROOT}/{relative}")
    info.size = len(data)
    info.mtime = 0
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    info.mode = 0o755 if executable else 0o644
    return info


def build_payload(project_root: Path, output: Path) -> dict[str, object]:
    root = Path(project_root)
    files = collect_payload_files(root)
    manifest_files = {
        path.relative_to(root).as_posix(): _sha256_file(path)
        for path in files
    }
    manifest = {
        "schema_version": "5.1",
        "archive_root": ARCHIVE_ROOT,
        "files": manifest_files,
    }
    manifest_bytes = (json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0, compresslevel=9) as zipped:
            with tarfile.open(fileobj=zipped, mode="w", format=tarfile.GNU_FORMAT) as archive:
                for path in files:
                    relative = path.relative_to(root).as_posix()
                    data = path.read_bytes()
                    executable = relative.startswith("scripts/v5/") and path.suffix == ".sh"
                    archive.addfile(_tar_info(relative, data, executable=executable), io.BytesIO(data))
                archive.addfile(
                    _tar_info("SHA256SUMS.json", manifest_bytes),
                    io.BytesIO(manifest_bytes),
                )

    return {
        "output": str(output),
        "files": len(files),
        "bytes": output.stat().st_size,
        "sha256": _sha256_file(output),
        "manifest_sha256": _sha256_bytes(manifest_bytes),
    }


def verify_payload(archive_path: Path) -> dict[str, object]:
    with tarfile.open(archive_path, "r:gz") as archive:
        manifest_member = archive.getmember(f"{ARCHIVE_ROOT}/SHA256SUMS.json")
        stream = archive.extractfile(manifest_member)
        if stream is None:
            raise ValueError("Payload manifest is unreadable")
        manifest = json.loads(stream.read().decode("utf-8"))
        checked = 0
        for relative, expected in manifest["files"].items():
            member = archive.getmember(f"{ARCHIVE_ROOT}/{relative}")
            file_stream = archive.extractfile(member)
            if file_stream is None:
                raise ValueError(f"Payload member is unreadable: {relative}")
            actual = hashlib.sha256(file_stream.read()).hexdigest()
            if actual != expected:
                raise ValueError(f"Payload hash mismatch: {relative}")
            checked += 1
    return {"status": "ok", "checked_files": checked, "sha256": _sha256_file(archive_path)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build deterministic PostTech Radar V5.1 GPU server payload")
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()

    result = build_payload(args.project_root, args.output)
    if args.verify:
        result["verification"] = verify_payload(args.output)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
