from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.v5.capture_environment import hash_model_snapshot  # noqa: E402


def completed_models(results_root: Path) -> list[dict[str, str]]:
    unique: dict[tuple[str, str], dict[str, str]] = {}
    for path in sorted(Path(results_root).glob("*/*/result.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if payload.get("status") != "complete":
            continue
        candidate = payload.get("candidate") or {}
        model_id = str(candidate.get("model_id") or "")
        revision = str(candidate.get("model_revision") or "")
        if model_id and revision:
            unique[(model_id, revision)] = {"model_id": model_id, "revision": revision}
    return [unique[key] for key in sorted(unique)]


def hash_completed_models(results_root: Path, output: Path) -> dict[str, Any]:
    from huggingface_hub import snapshot_download

    records = []
    for model in completed_models(results_root):
        snapshot = Path(
            snapshot_download(
                model["model_id"],
                revision=model["revision"],
                local_files_only=True,
            )
        )
        records.append({**model, **hash_model_snapshot(snapshot)})
    payload = {
        "schema_version": "5.1",
        "results_root": str(results_root),
        "models": records,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Hash immutable model snapshots used by completed V5 GPU runs")
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    payload = hash_completed_models(args.results_root, args.output)
    print(json.dumps({"output": str(args.output), "models": len(payload["models"])}, ensure_ascii=False))


if __name__ == "__main__":
    main()
