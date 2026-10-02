from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

from app.core.config import DATABASE_PATH  # noqa: E402
from app.ml.training import registration_text  # noqa: E402
from app.ml.v4.evaluation import safe_registration_row  # noqa: E402
from app.services.data_service import load_rows  # noqa: E402


def _bucket(request_id: str) -> int:
    return int(hashlib.sha256(f"v4-ood:{request_id}".encode()).hexdigest()[:8], 16) % 10


def _metrics(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, float]:
    fpr, tpr, _ = roc_curve(labels, scores)
    valid = np.flatnonzero(tpr >= 0.95)
    return {
        "auroc": round(float(roc_auc_score(labels, scores)), 6),
        "aupr": round(float(average_precision_score(labels, scores)), 6),
        "fpr_at_95_tpr": round(float(fpr[valid[0]]) if len(valid) else 1.0, 6),
        "known_false_unknown_rate": round(float((scores[labels == 0] >= threshold).mean()), 6),
        "ood_recall": round(float((scores[labels == 1] >= threshold).mean()), 6),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    pointer = json.loads((PROJECT_ROOT / "artifacts/gpu_research_v4/protocol.json").read_text())
    protocol = json.loads((PROJECT_ROOT / pointer["protocol_path"]).read_text())
    rows = [row for row in load_rows(DATABASE_PATH) if str(row.get("description") or "").strip()]
    ood_categories = list(protocol["ood_categories"])
    development_ood = set(ood_categories[:2])
    test_ood = set(ood_categories[2:])
    known = [row for row in rows if str(row["category"]) not in set(ood_categories)]
    train = [row for row in known if _bucket(str(row["request_id"])) < 7]
    calibration_known = [row for row in known if _bucket(str(row["request_id"])) in {7, 8}]
    test_known = [row for row in known if _bucket(str(row["request_id"])) == 9]
    calibration_ood = [row for row in rows if str(row["category"]) in development_ood]
    test_ood_rows = [row for row in rows if str(row["category"]) in test_ood]

    train_text = [registration_text(safe_registration_row(row)) for row in train]
    word = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=30000, sublinear_tf=True)
    char = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, max_features=40000)
    train_matrix = sparse.hstack([word.fit_transform(train_text), char.fit_transform(train_text)], format="csr")
    model = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=20260917)
    model.fit(train_matrix, [str(row["category"]) for row in train])

    def scores(items: list[dict[str, Any]]) -> np.ndarray:
        text = [registration_text(safe_registration_row(row)) for row in items]
        matrix = sparse.hstack([word.transform(text), char.transform(text)], format="csr")
        max_probability = model.predict_proba(matrix).max(axis=1)
        # Low classifier certainty is the deployment-cheap OOD signal.
        return 1.0 - np.asarray(max_probability)

    calibration_rows = [*calibration_known, *calibration_ood]
    calibration_labels = np.asarray(
        [0] * len(calibration_known) + [1] * len(calibration_ood), dtype=int
    )
    calibration_scores = scores(calibration_rows)
    thresholds = sorted(set(float(value) for value in calibration_scores))
    threshold = min(
        thresholds,
        key=lambda value: (
            -float((calibration_scores[calibration_labels == 1] >= value).mean())
            + 2 * float((calibration_scores[calibration_labels == 0] >= value).mean()),
            value,
        ),
    )
    test_rows = [*test_known, *test_ood_rows]
    test_labels = np.asarray([0] * len(test_known) + [1] * len(test_ood_rows), dtype=int)
    test_scores = scores(test_rows)
    payload = {
        "candidate_id": "ood/registration-tfidf-max-probability/v4",
        "status": "MEASURED_GROUP_CATEGORY_HOLDOUT",
        "dataset_sha256": protocol["dataset_sha256"],
        "split_sha256": protocol["split_sha256"],
        "registration_fields_only": True,
        "development_ood_categories": sorted(development_ood),
        "sealed_test_ood_categories": sorted(test_ood),
        "rows": {
            "train_known": len(train),
            "calibration_known": len(calibration_known),
            "calibration_ood": len(calibration_ood),
            "test_known": len(test_known),
            "test_ood": len(test_ood_rows),
        },
        "threshold": round(threshold, 8),
        "calibration": _metrics(calibration_labels, calibration_scores, threshold),
        "test": _metrics(test_labels, test_scores, threshold),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":
    main()
