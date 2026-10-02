from __future__ import annotations

import argparse
import hashlib
import shutil
from pathlib import Path

try:
    from project_setup import inspect_model_addons, project_root
except ModuleNotFoundError:  # imported as scripts.audit_submission in tests
    from scripts.project_setup import inspect_model_addons, project_root

EXPECTED_HASHES = {
    "models/customer_2026-09-21/category_lite_finetuned.joblib": "b8d2c02c5eef0f71bf6d5b62836718f6c064c25313a5428db3032788600c99bc",
    "models/customer_2026-09-21/category_qwen_finetuned.joblib": "447236e6627c808f5960c0368253e92a045e9bf77a3190d62504cd766afa3192",
    "models/customer_2026-09-21/minilm_supcon_s120/model.safetensors": "a4a489fc60bfc820678a44d606e084fa2ead4a751c588af8252a384977c94ab3",
    "models/customer_2026-09-21/qwen_r16_mnrl_s80_adapter/adapter_model.safetensors": "c6422a1b9eeb00583e419f211defcf25aa0074f63032afcfe38685f1e427b063",
    "models/category.joblib": "e72aa031463125cdb03050a34685967a5f7031709c8b385418cef8b64564cff5",
    "models/routing.joblib": "35cf99f8e05eabc5e113ba7bcdaa9ae9c39cef291a097b655f0950808d0ab532",
    "models/retrieval.joblib": "52cd19904edc269c31706345d0e26af7be671acc6f80add21c1260b2eca1267c",
    "data/raw/Обращения_1931.xlsx": "daf742da6bb5902149585a05a3e38a2c655347cf6001f9cae640e1f27069ac40",
    "data/raw/ALL_TICKETS.xlsx": "9611a2fb9d3334c1e4c946d36528218c7f75456542e524fea2f313d64c69a3d6",
}

EXCLUDED_PARTS = {
    ".git",
    ".pytest_cache",
    ".ruff_cache",
    "__pycache__",
    "node_modules",
    "dist",
    "encoders",
    "previous",
    "processed",
    "synthetic",
    "artifacts",
    "outputs",
    "backups",
    "work",
    "superpowers",
    ".superpowers",
}
EXCLUDED_NAMES = {".DS_Store", "README_FINAL.md", "GITLAB_UPLOAD_PROMPT.md"}
EXCLUDED_SUFFIXES = (".pyc", ".log", ".db", ".db-shm", ".db-wal", ".zip", ".tar.gz", ".tgz", ".tsbuildinfo")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def included(path: Path, root: Path) -> bool:
    relative = path.relative_to(root)
    if path.name in EXCLUDED_NAMES or any(part.startswith(".venv") for part in relative.parts):
        return False
    if any(part in EXCLUDED_PARTS for part in relative.parts):
        return False
    if relative.parts[:2] == ("scripts", "v4"):
        return False
    if path.name in {"vite.config.js", "vite.config.d.ts"} and "frontend" in relative.parts:
        return False
    if relative.match("models/v5/*.joblib") or relative.match("models/customer_2026-09-21/*.npz"):
        return False
    return not path.name.endswith(EXCLUDED_SUFFIXES)


def submission_files(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*") if path.is_file() and included(path, root))


def audit(root: Path) -> list[Path]:
    problems = inspect_model_addons(root)
    for relative, expected in EXPECTED_HASHES.items():
        path = root / relative
        if not path.is_file():
            problems.append(f"missing required file: {relative}")
        elif sha256(path) != expected:
            problems.append(f"unexpected SHA-256: {relative}")

    files = submission_files(root)
    for path in files:
        if path.stat().st_size > 500 * 1024 * 1024:
            problems.append(f"file is larger than 500 MiB: {path.relative_to(root)}")
        if path.stat().st_size <= 5 * 1024 * 1024 and path.resolve() != Path(__file__).resolve():
            content = path.read_bytes()
            if b"glpat-" in content or b"-----BEGIN PRIVATE KEY-----" in content:
                problems.append(f"possible secret in: {path.relative_to(root)}")
            if b"/Users/leonidbolbacan/" in content:
                problems.append(f"local absolute path in: {path.relative_to(root)}")

    if problems:
        raise SystemExit("Submission audit failed:\n- " + "\n- ".join(problems))
    return files


def export(files: list[Path], root: Path, destination: Path) -> None:
    if destination.exists():
        shutil.rmtree(destination)
    for source in files:
        target = destination / source.relative_to(root)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)


def main() -> None:
    parser = argparse.ArgumentParser(description="Проверить состав репозитория перед отправкой")
    parser.add_argument("--export", type=Path, help="создать чистую копию в указанной папке")
    args = parser.parse_args()

    root = project_root()
    files = audit(root)
    total = sum(path.stat().st_size for path in files)
    largest = sorted(files, key=lambda path: path.stat().st_size, reverse=True)[:5]
    print(f"Файлов в сдаче: {len(files)}")
    print(f"Общий размер: {total / 1024 / 1024:.1f} MiB")
    print("Самые крупные файлы:")
    for path in largest:
        print(f"  {path.stat().st_size / 1024 / 1024:8.1f} MiB  {path.relative_to(root)}")

    if args.export:
        destination = args.export.expanduser().resolve()
        export(files, root, destination)
        print(f"Чистая копия создана: {destination}")


if __name__ == "__main__":
    main()
