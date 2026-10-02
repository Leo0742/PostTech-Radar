from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

from app.core.config import DATABASE_PATH  # noqa: E402
from app.ml.v3.dataset import challenge_dataset_config, training_corpus_summary  # noqa: E402
from app.ml.v3.protocol import audit_manifest, build_v3_manifest  # noqa: E402
from app.services.data_service import load_rows  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Create or verify the sealed PostTech Radar v3 evaluation protocol")
    parser.add_argument("--database", type=Path, default=DATABASE_PATH)
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "artifacts/evaluation/v3/protocol.json")
    parser.add_argument("--seed", type=int, default=20260916)
    parser.add_argument("--top-k", type=int, default=15)
    parser.add_argument("--minimum-class-support", type=int, default=1)
    args = parser.parse_args()

    config = challenge_dataset_config(top_k=args.top_k, minimum_class_support=args.minimum_class_support)
    summary = training_corpus_summary(args.database, config)
    labels = {item["name"] for item in summary["top_k"]}
    rows = [row for row in load_rows(args.database) if row["category"] in labels and row["description"].strip()]
    manifest = build_v3_manifest(rows, seed=args.seed)
    payload = manifest.to_dict() | {
        "audit": audit_manifest(manifest),
        "dataset_policy": {
            "top_k": args.top_k,
            "minimum_class_support": args.minimum_class_support,
            "derived_from_complete_database": True,
            "eligible_records": len(rows),
            "database_records": summary["database_records"],
        },
        "sealed": True,
        "holdout_usage": "Evaluate a frozen shortlist once; never tune against these labels.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.output.exists():
        existing = json.loads(args.output.read_text(encoding="utf-8"))
        if existing.get("dataset_sha256") != payload["dataset_sha256"] or existing.get("split_sha256") != payload["split_sha256"]:
            raise SystemExit("Refusing to overwrite a different sealed v3 protocol")
        print(json.dumps({"status": "verified", "path": str(args.output), "split_sha256": payload["split_sha256"]}, indent=2))
        return
    args.output.write_text(serialized + "\n", encoding="utf-8")
    print(json.dumps({"status": "created", "path": str(args.output), "split_sha256": payload["split_sha256"]}, indent=2))


if __name__ == "__main__":
    main()
