from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, f1_score, log_loss

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

from app.core.config import DATABASE_PATH  # noqa: E402
from app.ml.v3.candidates import CandidateSpec, build_estimator, safe_feature_row  # noqa: E402
from app.ml.v3.conformal import APSConformalClassifier  # noqa: E402
from app.ml.v3.metrics import risk_coverage_curve  # noqa: E402
from app.ml.v3.protocol import manifest_from_dict  # noqa: E402
from app.ml.v3.selective import coverage_at_accuracy_targets, learn_class_thresholds  # noqa: E402
from app.services.data_service import load_rows  # noqa: E402


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _metrics(truth, probabilities, labels):
    predicted = np.asarray(labels)[probabilities.argmax(axis=1)]
    report = classification_report(truth, predicted, labels=labels, output_dict=True, zero_division=0)
    return {
        "accuracy": round(float(accuracy_score(truth, predicted)), 6),
        "macro_f1": round(float(f1_score(truth, predicted, average="macro", zero_division=0)), 6),
        "weighted_f1": round(float(f1_score(truth, predicted, average="weighted", zero_division=0)), 6),
        "nll": round(float(log_loss(truth, probabilities, labels=labels)), 6),
        "worst_class_f1": round(min(float(report[label]["f1-score"]) for label in labels), 6),
        "per_class": {
            label: {
                "precision": round(float(report[label]["precision"]), 6),
                "recall": round(float(report[label]["recall"]), 6),
                "f1": round(float(report[label]["f1-score"]), 6),
                "support": int(report[label]["support"]),
            }
            for label in labels
        },
    }


def main() -> None:
    final_path = PROJECT_ROOT / "artifacts/evaluation/v3/final_routing_holdout.json"
    if final_path.exists():
        print(json.dumps({"status": "REFUSED_ALREADY_EVALUATED", "artifact": str(final_path)}, ensure_ascii=False))
        return
    allowed = {"(1 линия)", "(2 линия)", "(3 линия)"}
    rows = [
        {**row, "category": row["final_line"]}
        for row in load_rows(DATABASE_PATH)
        if row["final_line"] in allowed and str(row.get("description") or "").strip()
    ]
    by_id = {str(row["request_id"]): row for row in rows}
    manifest = manifest_from_dict(json.loads((PROJECT_ROOT / "artifacts/evaluation/v3/protocol_routing.json").read_text(encoding="utf-8")))
    labels = list(manifest.labels)
    development = [by_id[value] for value in manifest.development_request_ids]
    calibration = [by_id[value] for value in manifest.calibration_request_ids]
    holdout = [by_id[value] for value in manifest.holdout_request_ids]
    spec = CandidateSpec("routing/structured_lr/final", "routing", "structured_lr", "combined")
    model = build_estimator(spec)
    model.fit([safe_feature_row(row) for row in development], [row["category"] for row in development])
    positions = {str(label): index for index, label in enumerate(model.classes_)}
    def predict(selected):
        raw = model.predict_proba([safe_feature_row(row) for row in selected])
        return np.asarray([[values[positions[label]] for label in labels] for values in raw])
    calibration_probabilities = predict(calibration)
    holdout_probabilities = predict(holdout)
    calibration_truth = [row["category"] for row in calibration]
    holdout_truth = [row["category"] for row in holdout]
    thresholds = learn_class_thresholds(calibration_truth, calibration_probabilities, labels, target_accuracy=0.95, minimum_predictions=3)
    predictions = np.asarray(labels)[holdout_probabilities.argmax(axis=1)]
    confidence = holdout_probabilities.max(axis=1)
    accepted = np.asarray([confidence[index] >= thresholds[label] for index, label in enumerate(predictions)])
    conformal = []
    for alpha in (0.1, 0.05, 0.025, 0.01):
        estimator = APSConformalClassifier(alpha=alpha).fit(calibration_truth, calibration_probabilities, labels)
        conformal.append(estimator.summary(holdout_truth, holdout_probabilities))
    payload = {
        "sealed_holdout_opened_once": True,
        "winner": "structured word+char TF-IDF + separately encoded registration metadata + LogisticRegression",
        "development_cv": json.loads((PROJECT_ROOT / "artifacts/research/v3/candidates/routing-structured_lr.json").read_text(encoding="utf-8"))["metrics"],
        "holdout": {
            "size": len(holdout),
            "metrics": _metrics(holdout_truth, holdout_probabilities, labels),
            "risk_coverage": risk_coverage_curve(holdout_truth, holdout_probabilities, labels),
            "coverage_at_accuracy_targets": coverage_at_accuracy_targets(holdout_truth, holdout_probabilities, labels),
            "class_specific_thresholds_95": {
                "coverage": round(float(accepted.mean()), 6),
                "accepted_accuracy": round(float((predictions[accepted] == np.asarray(holdout_truth)[accepted]).mean()), 6) if accepted.any() else 0.0,
                "errors": int((predictions[accepted] != np.asarray(holdout_truth)[accepted]).sum()),
            },
            "conformal": conformal,
        },
        "line_4": "UNTRAINABLE_SINGLE_EXAMPLE_PRESERVED_AS_MANUAL_REVIEW",
    }
    _write(final_path, payload)
    deployment = build_estimator(CandidateSpec("routing/structured_lr/deployment", "routing", "structured_lr", "combined"))
    deployment.fit([safe_feature_row(row) for row in rows], [row["category"] for row in rows])
    model_dir = PROJECT_ROOT / "artifacts/models/v3/routing-structured"
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(deployment, model_dir / "model.joblib")
    _write(model_dir / "metadata.json", {"training_count": len(rows), "labels": labels, "class_thresholds": thresholds})
    print(json.dumps(payload["holdout"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
