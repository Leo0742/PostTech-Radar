from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.optimize import minimize_scalar
from sklearn.metrics import log_loss
from sklearn.svm import LinearSVC

ROOT = Path(__file__).resolve().parents[1]
for import_path in (ROOT, ROOT / "backend"):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from scripts.calibrate_lite_v2_specialist import (  # noqa: E402
    SPECIALIST_CLASSIFIER,
    SPECIALIST_FEATURE,
    _aligned_scores,
    _apply_specialist,
    _policy,
    _softmax,
)
from scripts.lite_v2_feature_tune import Recipe as TextRecipe  # noqa: E402
from scripts.lite_v2_feature_tune import make_matrices  # noqa: E402
from scripts.lite_v2_sprint import classifier, load_folds, matrices  # noqa: E402
from scripts.lite_v3_local_max import (  # noqa: E402
    ensure_minilm_embeddings,
    fit_qwen_projection,
    load_qwen_teacher_embeddings,
)
from scripts.v5_dataset import load_v5_rows  # noqa: E402
from scripts.v5_experiment import DEFAULT_CONTRACT, labels_for_view, load_json  # noqa: E402

SEED = 20260920


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, default=ROOT / "outputs/lite_v3/minilm_256.npy")
    parser.add_argument(
        "--qwen-artifact",
        type=Path,
        default=ROOT / "models/v5/category_qwen4b_lite.joblib",
    )
    parser.add_argument(
        "--specialist-validation",
        type=Path,
        default=ROOT / "outputs/lite_v3/qdistill_specialist.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/lite_v3/calibration_qdistill_specialist.json",
    )
    args = parser.parse_args()

    specialist_validation = json.loads(args.specialist_validation.read_text(encoding="utf-8"))["specialist"]
    pair_set = {frozenset(map(str, pair)) for pair in specialist_validation["deployment_pairs"]}
    specialist_threshold = float(specialist_validation["deployment_threshold"])

    rows = load_v5_rows()
    labels = labels_for_view(load_json(DEFAULT_CONTRACT), "full43")
    label_index = {label: index for index, label in enumerate(labels)}
    student = ensure_minilm_embeddings(rows, args.cache, device="mps", batch_size=32, max_seq_length=256)
    teacher = load_qwen_teacher_embeddings(args.qwen_artifact, rows)
    teacher = teacher / np.maximum(np.linalg.norm(teacher, axis=1, keepdims=True), 1e-12)
    row_index = {str(row["request_id"]): index for index, row in enumerate(rows)}
    text_recipe = TextRecipe("word11", (1, 1), (3, 5))

    all_scores = []
    all_specialist_scores = []
    truth = []
    started = time.perf_counter()
    for fold_number, (repeat, fold, train_rows, validation_rows) in enumerate(load_folds(rows), 1):
        print(f"qdistill calibration fold {fold_number}/12 repeat={repeat} fold={fold}", flush=True)
        y_train = [str(row["category"]) for row in train_rows]
        y_validation = [str(row["category"]) for row in validation_rows]
        train_indices = [row_index[str(row["request_id"])] for row in train_rows]
        validation_indices = [row_index[str(row["request_id"])] for row in validation_rows]

        projected_train, projected_validation = fit_qwen_projection(
            student[train_indices],
            teacher[train_indices],
            student[validation_indices],
            alpha=1.0,
        )
        x_train, x_validation = make_matrices(train_rows, validation_rows, text_recipe)
        fused_train = sparse.hstack([x_train, sparse.csr_matrix(projected_train)], format="csr")
        fused_validation = sparse.hstack(
            [x_validation, sparse.csr_matrix(projected_validation)], format="csr"
        )
        base = LinearSVC(
            C=0.35,
            class_weight="balanced",
            max_iter=10000,
            random_state=SEED + repeat * 10 + fold,
        ).fit(fused_train, y_train)
        all_scores.append(_aligned_scores(base, fused_validation, labels))

        s_train, s_validation = matrices(train_rows, validation_rows, SPECIALIST_FEATURE)
        specialist = classifier(SPECIALIST_CLASSIFIER).fit(s_train, y_train)
        all_specialist_scores.append(_aligned_scores(specialist, s_validation, labels))
        truth.extend(label_index[label] for label in y_validation)

    scores = np.vstack(all_scores)
    specialist_scores = np.vstack(all_specialist_scores)
    truth_array = np.asarray(truth, dtype=int)

    def objective(log_temperature: float) -> float:
        temperature = math.exp(float(log_temperature))
        probabilities = _apply_specialist(
            _softmax(scores, temperature),
            specialist_scores,
            labels,
            pair_set,
            specialist_threshold,
        )
        return float(log_loss(truth_array, probabilities, labels=np.arange(len(labels))))

    optimized = minimize_scalar(
        objective,
        bounds=(math.log(0.03), math.log(2.0)),
        method="bounded",
        options={"xatol": 1e-6},
    )
    temperature = math.exp(float(optimized.x))
    probabilities = _apply_specialist(
        _softmax(scores, temperature),
        specialist_scores,
        labels,
        pair_set,
        specialist_threshold,
    )
    policy = _policy(probabilities, truth_array)
    top1 = np.argmax(probabilities, axis=1)
    payload = {
        "validation": "12 frozen grouped DEVELOPMENT folds; Qwen projection and specialist both fit inside each outer TRAIN fold",
        "lockbox_accessed": False,
        "rows_oof": int(len(truth_array)),
        "temperature": round(float(temperature), 6),
        "nll": round(float(objective(math.log(temperature))), 6),
        "top1_accuracy": round(float((top1 == truth_array).mean()), 6),
        "threshold": policy["threshold"],
        "margin_threshold": policy["margin_threshold"],
        "policy_objective": "maximize F1 for OOF top1-correctness detection; tie-break accepted accuracy then coverage",
        "correctness_f1": round(float(policy["correctness_f1"]), 6),
        "coverage": round(float(policy["coverage"]), 6),
        "accepted_accuracy": round(float(policy["accepted_accuracy"]), 6),
        "review_rate": round(float(policy["review_rate"]), 6),
        "review_bucket_error_rate": round(float(policy["review_bucket_error_rate"]), 6),
        "specialist_pairs": specialist_validation["deployment_pairs"],
        "specialist_threshold": specialist_threshold,
        "quality_note": "Use qdistill_specialist.json cross-repeat metrics for the honest quality claim.",
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
