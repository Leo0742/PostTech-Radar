from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

from app.core.config import DATABASE_PATH  # noqa: E402
from app.ml.v3.candidates import CandidateSpec, build_estimator, safe_feature_row  # noqa: E402
from app.ml.v4.evaluation import classification_metrics  # noqa: E402
from app.services.data_service import load_rows  # noqa: E402


def _fit_predict(train: list[dict[str, Any]], test: list[dict[str, Any]], candidate_id: str) -> tuple[Any, list[str]]:
    model = build_estimator(CandidateSpec(candidate_id, "category", "structured_lr", "combined"))
    model.fit([safe_feature_row(row) for row in train], [str(row["category"]) for row in train])
    return model, [str(value) for value in model.predict([safe_feature_row(row) for row in test])]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--models-dir", type=Path, default=PROJECT_ROOT / "models")
    args = parser.parse_args()
    pointer = json.loads((PROJECT_ROOT / "artifacts/gpu_research_v4/protocol.json").read_text())
    protocol = json.loads((PROJECT_ROOT / pointer["protocol_path"]).read_text())
    rows = [row for row in load_rows(DATABASE_PATH) if str(row.get("description") or "").strip()]
    by_id = {str(row["request_id"]): row for row in rows}
    development_ids = set(protocol["development_request_ids"])
    calibration_ids = set(protocol["calibration_request_ids"])
    holdout_ids = set(protocol["holdout_request_ids"])
    train_ids = development_ids | calibration_ids
    counts = Counter(str(row["category"]) for row in rows)
    top15 = [label for label, _ in counts.most_common(15)]
    reports: dict[str, Any] = {}
    started = time.perf_counter()
    for view, view_labels in (("top15", set(top15)), ("full43", set(counts))):
        train = [by_id[item] for item in sorted(train_ids) if item in by_id and str(by_id[item]["category"]) in view_labels]
        test = [by_id[item] for item in sorted(holdout_ids) if item in by_id and str(by_id[item]["category"]) in view_labels]
        _, predicted = _fit_predict(train, test, f"category/structured_lr/{view}/sealed-v4")
        reports[view] = {
            "train_rows": len(train),
            "holdout_rows": len(test),
            "metrics": classification_metrics(
                [str(row["category"]) for row in test], predicted, labels=sorted(view_labels)
            ),
        }

    temporal_train = [by_id[item] for item in protocol["temporal_train_request_ids"] if item in by_id]
    temporal_test = [by_id[item] for item in protocol["temporal_test_request_ids"] if item in by_id]
    temporal_train_labels = {str(row["category"]) for row in temporal_train}
    temporal_test_evaluable = [row for row in temporal_test if str(row["category"]) in temporal_train_labels]
    _, temporal_predicted = _fit_predict(
        temporal_train, temporal_test_evaluable, "category/structured_lr/temporal-v4"
    )
    reports["temporal"] = {
        "train_rows": len(temporal_train),
        "test_rows": len(temporal_test),
        "evaluable_test_rows": len(temporal_test_evaluable),
        "unseen_label_rows": len(temporal_test) - len(temporal_test_evaluable),
        "metrics": classification_metrics(
            [str(row["category"]) for row in temporal_test_evaluable],
            temporal_predicted,
            labels=sorted(set(counts)),
        ),
    }

    deployment, _ = _fit_predict(rows, rows[:1], "category/structured_lr/deployment-v4")
    v4_dir = args.models_dir / "v4"
    v4_dir.mkdir(parents=True, exist_ok=True)
    model_path = v4_dir / "category.joblib"
    trained_at = datetime.now(UTC).isoformat()
    bundle = {
        "pipeline": deployment,
        "explanation_pipeline": None,
        "input_contract": "registration_mapping",
        "top15": top15,
        "threshold": 0.09951064,
        "margin_threshold": 0.0,
        "metadata": {
            "candidate_id": "category/structured_lr/full43/deployment-v4",
            "family": "structured_word_char_tfidf_logistic_regression",
            "dataset_sha256": protocol["dataset_sha256"],
            "split_sha256": protocol["split_sha256"],
            "training_rows": len(rows),
            "features": list(safe_feature_row(rows[0])),
            "trained_at": trained_at,
        },
    }
    joblib.dump(bundle, model_path)
    artifact_sha256 = hashlib.sha256(model_path.read_bytes()).hexdigest()
    bundle["metadata"]["artifact_sha256"] = artifact_sha256
    joblib.dump(bundle, model_path)
    artifact_sha256 = hashlib.sha256(model_path.read_bytes()).hexdigest()
    champion = {
        **bundle["metadata"],
        "artifact_sha256": artifact_sha256,
        "artifact": str(model_path.relative_to(PROJECT_ROOT)),
        "license": "project-trained/scikit-learn",
        "holdout": reports,
    }
    registry = {
        "ACTIVE_CHAMPION": champion,
        "PREVIOUS_CHAMPION": {"candidate_id": "category/tuned_word_char_lr/v3", "artifact": "models/category.joblib"},
        "CHALLENGERS": [
            "category/oof-stack/top15/v4",
            "category/bge-m3-frozen/top15/v4",
            "category/qwen3-embedding-06b-frozen/top15/v4",
        ],
    }
    (v4_dir / "registry.json").write_text(json.dumps(registry, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    policy = {
        "unknown_label": "UNKNOWN_NEW_ISSUE",
        "known_threshold": bundle["threshold"],
        "champion": champion,
    }
    (args.models_dir / "v4_runtime.json").write_text(
        json.dumps(policy, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    payload = {
        "candidate_id": champion["candidate_id"],
        "status": "SEALED_EVALUATED_AND_EXPORTED",
        "sealed_holdout_accessed": True,
        "real_only_evaluation": True,
        "dataset_sha256": protocol["dataset_sha256"],
        "split_sha256": protocol["split_sha256"],
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "reports": reports,
        "registry": registry,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "reports": reports, "artifact": str(model_path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
