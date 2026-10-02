from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from app.core.config import DATABASE_PATH  # noqa: E402
from app.ml.continuous_learning import build_retraining_dataset, train_challenger  # noqa: E402
from app.ml.model_registry import ModelRegistry  # noqa: E402

CHAMPION_VERSION = "v5.2"
CHAMPION_METRICS = {
    "category_top1": 0.79883,
    "category_top3": 0.97136,
    "category_macro_f1": 0.74007,
    "full43_top1": 0.68171,
    "full43_top3": 0.88799,
    "full43_macro_f1": 0.53771,
}
CHAMPION_MODEL_ID = "Qwen/Qwen3-Embedding-8B"
CHAMPION_MODEL_REVISION = "1d8ad4ca9b3dd8059ad90a75d4983776a23d44af"


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _dataset_summary(dataset: dict[str, object]) -> dict[str, object]:
    canonical = json.dumps(dataset["rows"], ensure_ascii=False, sort_keys=True).encode("utf-8")
    return {
        "schema_version": dataset["schema_version"],
        "feature_contract": dataset["feature_contract"],
        "historical_count": dataset["historical_count"],
        "feedback_count": dataset["feedback_count"],
        "rows_total": dataset["rows_total"],
        "dataset_sha256": _sha256_bytes(canonical),
    }


def _write_report(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _ensure_champion(registry: ModelRegistry) -> None:
    if registry.champion() is not None:
        return
    registry.register(
        {
            "version": CHAMPION_VERSION,
            "model_id": CHAMPION_MODEL_ID,
            "model_revision": CHAMPION_MODEL_REVISION,
            "metrics": CHAMPION_METRICS,
            "historical_row_count": 1931,
            "feedback_row_count": 0,
            "feature_contract": "registration_mapping",
            "status": "champion",
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Controlled PostTech Radar V6 retraining pipeline")
    parser.add_argument("--database", type=Path, default=DATABASE_PATH)
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs" / "retraining")
    parser.add_argument("--registry", type=Path, default=ROOT / "models" / "model_registry.json")
    parser.add_argument("--models-root", type=Path, default=ROOT / "models" / "challengers")
    parser.add_argument("--version", default=datetime.now(UTC).strftime("v6.0-%Y%m%d-%H%M%S"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--promote", metavar="VERSION")
    args = parser.parse_args()

    registry = ModelRegistry(args.registry)
    if args.promote:
        promoted = registry.promote(args.promote)
        print(json.dumps({"promoted": promoted, "automatic_deployment": False}, ensure_ascii=False, indent=2))
        return

    dataset = build_retraining_dataset(args.database)
    summary = _dataset_summary(dataset)
    if args.dry_run:
        payload = {
            "version": args.version,
            "mode": "dry-run",
            "dataset": summary,
            "trained": False,
            "registered": False,
            "automatic_deployment": False,
        }
        _write_report(args.output_root / f"{args.version}.json", payload)
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    _ensure_champion(registry)
    artifact = args.models_root / args.version / "challenger.joblib"
    metrics = train_challenger(dataset["rows"], artifact)
    artifact_hash = _sha256_file(artifact)
    trained_at = datetime.now(UTC).isoformat()
    record = registry.register(
        {
            "version": args.version,
            "model_id": "posttech/tfidf-controlled-challenger",
            "model_revision": "local-v6",
            "dataset_hash": summary["dataset_sha256"],
            "historical_row_count": summary["historical_count"],
            "feedback_row_count": summary["feedback_count"],
            "metrics": metrics,
            "training_timestamp": trained_at,
            "feature_contract": summary["feature_contract"],
            "artifact_hashes": {str(artifact.relative_to(ROOT)): artifact_hash},
            "status": "challenger",
        }
    )
    payload = {
        "version": args.version,
        "mode": "train-challenger",
        "dataset": summary,
        "challenger": record,
        "champion": registry.champion(),
        "automatic_deployment": False,
        "promotion_command": f"python scripts/retrain_v6.py --registry {args.registry} --promote {args.version}",
    }
    _write_report(args.output_root / f"{args.version}.json", payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
