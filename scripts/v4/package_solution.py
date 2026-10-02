from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "outputs/PostTech_Radar_v4_complete.zip"
EXCLUDED_PARTS = {
    ".git",
    ".pytest_cache",
    ".ruff_cache",
    ".pnpm-store",
    ".presentation-build",
    ".codex-finalizer",
    "node_modules",
    "__pycache__",
    "tmp",
    "work",
    "logs",
}
EXCLUDED_PREFIXES = (".venv",)


def _included(path: Path) -> bool:
    relative = path.relative_to(ROOT)
    if any(part in EXCLUDED_PARTS or part.startswith(EXCLUDED_PREFIXES) for part in relative.parts):
        return False
    if path == OUTPUT or path.suffix == ".zip":
        return False
    if "embedding-cache" in relative.parts or path.name.endswith(".embeddings.npz"):
        return False
    # User requested deletion of downloaded research models. Keep the trained
    # classifier head/registry, but do not put third-party encoder weights in the archive.
    if relative.parts[:3] == ("models", "v4", "encoder"):
        return False
    return path.is_file()


def main() -> None:
    files = sorted(path for path in ROOT.rglob("*") if _included(path))
    manifest = {
        "archive": str(OUTPUT.name),
        "files": len(files),
        "excluded": [
            "Python virtual environments",
            "node_modules/pnpm cache",
            "temporary work/log directories",
            "downloaded third-party encoder weights and model caches",
            "recomputable embedding matrices/caches",
            "older ZIP archives",
        ],
        "runtime_note": (
            "models/v4/category.joblib and the pinned model registry are included. "
            "If the selected encoder weights are absent, runtime attempts the pinned model ID/revision and then uses the bundled v3 fallback on inference failure."
        ),
    }
    manifest_path = ROOT / "outputs/PACKAGE_CONTENTS_V4.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if manifest_path not in files:
        files.append(manifest_path)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(OUTPUT, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in files:
            archive.write(path, path.relative_to(ROOT))
    digest = hashlib.sha256(OUTPUT.read_bytes()).hexdigest()
    checksum = ROOT / "outputs/PostTech_Radar_v4_complete.zip.sha256"
    checksum.write_text(f"{digest}  {OUTPUT.name}\n", encoding="utf-8")
    print(json.dumps({"archive": str(OUTPUT), "bytes": OUTPUT.stat().st_size, "sha256": digest}, ensure_ascii=False))


if __name__ == "__main__":
    main()
