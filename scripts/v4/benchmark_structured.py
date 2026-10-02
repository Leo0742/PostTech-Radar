from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

from app.core.config import DATABASE_PATH  # noqa: E402
from app.ml.v3.candidates import CandidateSpec, build_estimator, safe_feature_row  # noqa: E402
from app.ml.v4.evaluation import classification_metrics  # noqa: E402
from app.services.data_service import load_rows  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--view", choices=("top15", "full43"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    pointer = json.loads((PROJECT_ROOT / "artifacts/gpu_research_v4/protocol.json").read_text())
    protocol: dict[str, Any] = json.loads((PROJECT_ROOT / pointer["protocol_path"]).read_text())
    all_rows = [row for row in load_rows(DATABASE_PATH) if str(row.get("description") or "").strip()]
    counts = Counter(str(row["category"]) for row in all_rows)
    labels = {label for label, _ in counts.most_common(15)} if args.view == "top15" else set(counts)
    development = set(protocol["development_request_ids"])
    rows = [row for row in all_rows if str(row["request_id"]) in development and str(row["category"]) in labels]
    by_id = {str(row["request_id"]): row for row in rows}
    predictions = []
    folds = []
    fit_seconds = 0.0
    predict_seconds = 0.0
    for fold in protocol["folds"]:
        if int(fold["repeat"]) != 0:
            continue
        train = [by_id[item] for item in fold["train_request_ids"] if item in by_id]
        validation = [by_id[item] for item in fold["validation_request_ids"] if item in by_id]
        estimator = build_estimator(
            CandidateSpec(f"category/structured_lr/{args.view}/v4", "category", "structured_lr", "combined")
        )
        started = time.perf_counter()
        estimator.fit([safe_feature_row(row) for row in train], [str(row["category"]) for row in train])
        fit_seconds += time.perf_counter() - started
        started = time.perf_counter()
        values = estimator.predict([safe_feature_row(row) for row in validation])
        predict_seconds += time.perf_counter() - started
        predictions.extend(
            {
                "request_id": str(row["request_id"]),
                "truth": str(row["category"]),
                "prediction": str(value),
            }
            for row, value in zip(validation, values, strict=True)
        )
        folds.append({"fold": fold["fold"], "train": len(train), "validation": len(validation)})
    truth = [row["truth"] for row in predictions]
    predicted = [row["prediction"] for row in predictions]
    payload = {
        "candidate_id": f"category/structured_lr/{args.view}/v4",
        "status": "MEASURED_STRICT_OOF",
        "view": args.view,
        "dataset_sha256": protocol["dataset_sha256"],
        "split_sha256": protocol["split_sha256"],
        "registration_fields_only": True,
        "real_only_evaluation": True,
        "sealed_holdout_accessed": False,
        "fit_seconds": round(fit_seconds, 3),
        "prediction_ms_per_ticket": round(1000 * predict_seconds / max(1, len(predictions)), 4),
        "folds": folds,
        "metrics": classification_metrics(truth, predicted, labels=sorted(labels)),
        "predictions": predictions,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "metrics": payload["metrics"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
