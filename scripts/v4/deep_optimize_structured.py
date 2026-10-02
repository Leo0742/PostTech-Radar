from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

from app.core.config import DATABASE_PATH  # noqa: E402
from app.ml.v3.candidates import CandidateSpec, build_estimator, safe_feature_row  # noqa: E402
from app.ml.v4.evaluation import classification_metrics  # noqa: E402
from app.services.data_service import load_rows  # noqa: E402


def _objective(metrics: dict[str, Any], view: str) -> float:
    if view == "top15":
        return metrics["macro_f1"] + 0.25 * metrics["accuracy"] + 0.15 * metrics["worst_class_f1"]
    bands = metrics["support_bands"]
    return (
        metrics["macro_f1"]
        + 0.25 * metrics["balanced_accuracy"]
        + 0.2 * (bands["1-5"]["macro_f1"] or 0.0)
        + 0.1 * (bands["5-20"]["macro_f1"] or 0.0)
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--view", choices=("top15", "full43"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    pointer = json.loads((PROJECT_ROOT / "artifacts/gpu_research_v4/protocol.json").read_text())
    protocol = json.loads((PROJECT_ROOT / pointer["protocol_path"]).read_text())
    all_rows = [row for row in load_rows(DATABASE_PATH) if str(row.get("description") or "").strip()]
    counts = Counter(str(row["category"]) for row in all_rows)
    view_labels = {label for label, _ in counts.most_common(15)} if args.view == "top15" else set(counts)
    rows = [row for row in all_rows if str(row["category"]) in view_labels]
    by_id = {str(row["request_id"]): row for row in rows}
    grid = [
        {"C": c, "class_weight": weight, "max_iter": 2500}
        for c in (0.25, 0.5, 1.0, 2.0, 4.0, 8.0)
        for weight in (None, "balanced")
    ]
    tuning = []
    tuning_fold = next(
        fold for fold in protocol["folds"] if int(fold["repeat"]) == 0 and int(fold["fold"]) == 0
    )
    tune_train = [by_id[item] for item in tuning_fold["train_request_ids"] if item in by_id]
    tune_validation = [by_id[item] for item in tuning_fold["validation_request_ids"] if item in by_id]
    for config in grid:
        model = build_estimator(
            CandidateSpec("category/structured-deep/v4", "category", "structured_lr", "combined", parameters=config)
        )
        model.fit([safe_feature_row(row) for row in tune_train], [str(row["category"]) for row in tune_train])
        predicted = [str(value) for value in model.predict([safe_feature_row(row) for row in tune_validation])]
        metrics = classification_metrics(
            [str(row["category"]) for row in tune_validation], predicted, labels=sorted(view_labels)
        )
        tuning.append({"parameters": config, "objective": round(_objective(metrics, args.view), 8), "metrics": metrics})
    selected = max(tuning, key=lambda row: row["objective"])
    repeat_metrics = {}
    predictions = []
    for repeat in (0, 1, 2):
        truth: list[str] = []
        predicted: list[str] = []
        request_ids: list[str] = []
        for fold in protocol["folds"]:
            if int(fold["repeat"]) != repeat:
                continue
            train = [by_id[item] for item in fold["train_request_ids"] if item in by_id]
            validation = [by_id[item] for item in fold["validation_request_ids"] if item in by_id]
            parameters = {**selected["parameters"], "seed": 20260917 + repeat}
            model = build_estimator(
                CandidateSpec("category/structured-deep/v4", "category", "structured_lr", "combined", parameters=parameters)
            )
            model.fit([safe_feature_row(row) for row in train], [str(row["category"]) for row in train])
            values = [str(value) for value in model.predict([safe_feature_row(row) for row in validation])]
            truth.extend(str(row["category"]) for row in validation)
            predicted.extend(values)
            request_ids.extend(str(row["request_id"]) for row in validation)
        repeat_metrics[repeat] = classification_metrics(truth, predicted, labels=sorted(view_labels))
        predictions.extend(
            {"repeat": repeat, "request_id": item, "truth": target, "prediction": value}
            for item, target, value in zip(request_ids, truth, predicted, strict=True)
        )
    summary = defaultdict(list)
    for metrics in repeat_metrics.values():
        for key in ("accuracy", "macro_f1", "weighted_f1", "balanced_accuracy", "worst_class_f1"):
            summary[key].append(metrics[key])
    stability = {
        key: {"mean": round(float(np.mean(values)), 6), "std": round(float(np.std(values)), 6)}
        for key, values in summary.items()
    }
    temporal_train = [by_id[item] for item in protocol["temporal_train_request_ids"] if item in by_id]
    temporal_known = {str(row["category"]) for row in temporal_train}
    temporal_test = [
        by_id[item]
        for item in protocol["temporal_test_request_ids"]
        if item in by_id and str(by_id[item]["category"]) in temporal_known
    ]
    temporal_parameters = {**selected["parameters"], "seed": 20260920}
    temporal_model = build_estimator(
        CandidateSpec(
            "category/structured-deep/temporal-v4",
            "category",
            "structured_lr",
            "combined",
            parameters=temporal_parameters,
        )
    )
    temporal_model.fit(
        [safe_feature_row(row) for row in temporal_train], [str(row["category"]) for row in temporal_train]
    )
    temporal_predictions = [
        str(value) for value in temporal_model.predict([safe_feature_row(row) for row in temporal_test])
    ]
    temporal = {
        "train_rows": len(temporal_train),
        "test_rows": len(temporal_test),
        "metrics": classification_metrics(
            [str(row["category"]) for row in temporal_test], temporal_predictions, labels=sorted(view_labels)
        ),
    }
    payload = {
        "candidate_id": f"category/structured-deep/{args.view}/v4",
        "status": "MEASURED_DEEP_OPTIMIZATION",
        "view": args.view,
        "dataset_sha256": protocol["dataset_sha256"],
        "split_sha256": protocol["split_sha256"],
        "sealed_holdout_accessed": False,
        "real_only_evaluation": True,
        "grid_size": len(grid),
        "selected": selected,
        "repeat_metrics": repeat_metrics,
        "stability": stability,
        "temporal": temporal,
        "tuning": tuning,
        "predictions": predictions,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "selected": selected["parameters"], "stability": stability}, ensure_ascii=False))


if __name__ == "__main__":
    main()
