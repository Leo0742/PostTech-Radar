from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.metrics import f1_score, log_loss

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "backend"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from scripts.lite_v2_sprint import (  # noqa: E402
    ClassifierRecipe,
    FeatureRecipe,
    classifier,
    load_folds,
    matrices,
)
from scripts.v5_dataset import load_v5_rows  # noqa: E402
from scripts.v5_experiment import DEFAULT_CONTRACT, labels_for_view, load_json  # noqa: E402

BASE_FEATURE = FeatureRecipe(
    "lite_v2_final",
    "specialist",
    30000,
    60000,
    (1, 1),
    (3, 5),
    True,
    0.35,
)
BASE_CLASSIFIER = ClassifierRecipe("svc_c035_bal", "svc", 0.35, "balanced")
SPECIALIST_FEATURE = FeatureRecipe(
    "lite_v2_specialist",
    "specialist",
    30000,
    60000,
    (1, 2),
    (3, 5),
    False,
    1.0,
)
SPECIALIST_CLASSIFIER = ClassifierRecipe("svc_c10_bal", "svc", 1.0, "balanced")


def _aligned_scores(model: Any, matrix: Any, labels: list[str]) -> np.ndarray:
    local = np.asarray(model.decision_function(matrix), dtype=float)
    if local.ndim == 1:
        local = np.column_stack([-local, local])
    result = np.full((matrix.shape[0], len(labels)), -1e9, dtype=float)
    positions = {label: index for index, label in enumerate(labels)}
    for source, label in enumerate(model.classes_):
        target = positions.get(str(label))
        if target is not None:
            result[:, target] = local[:, source]
    return result


def _softmax(scores: np.ndarray, temperature: float) -> np.ndarray:
    scaled = np.asarray(scores, dtype=float) / max(float(temperature), 1e-6)
    scaled -= np.max(scaled, axis=1, keepdims=True)
    exp = np.exp(scaled)
    return exp / exp.sum(axis=1, keepdims=True)


def _apply_specialist(
    probabilities: np.ndarray,
    specialist_scores: np.ndarray,
    labels: list[str],
    pairs: set[frozenset[str]],
    threshold: float,
) -> np.ndarray:
    result = np.asarray(probabilities, dtype=float).copy()
    for row_index in range(len(result)):
        order = np.argsort(-result[row_index])
        if len(order) < 2:
            continue
        first_index, second_index = int(order[0]), int(order[1])
        first, second = labels[first_index], labels[second_index]
        if frozenset((first, second)) not in pairs:
            continue
        margin = specialist_scores[row_index, second_index] - specialist_scores[row_index, first_index]
        if margin > float(threshold):
            result[row_index, first_index], result[row_index, second_index] = (
                result[row_index, second_index],
                result[row_index, first_index],
            )
    return result


def _policy(probabilities: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    order = np.argsort(-probabilities, axis=1)
    predictions = order[:, 0]
    correct = predictions == truth
    confidence = probabilities[np.arange(len(probabilities)), order[:, 0]]
    margin = confidence - probabilities[np.arange(len(probabilities)), order[:, 1]]

    best: dict[str, float] | None = None
    for threshold in np.arange(0.10, 0.901, 0.01):
        for margin_threshold in np.arange(0.0, 0.301, 0.01):
            accepted = (confidence >= threshold) & (margin >= margin_threshold)
            score = float(f1_score(correct, accepted, zero_division=0))
            coverage = float(accepted.mean())
            accepted_accuracy = float(correct[accepted].mean()) if accepted.any() else 0.0
            review = ~accepted
            review_error_rate = float((~correct[review]).mean()) if review.any() else 0.0
            candidate = {
                "threshold": round(float(threshold), 2),
                "margin_threshold": round(float(margin_threshold), 2),
                "correctness_f1": score,
                "coverage": coverage,
                "accepted_accuracy": accepted_accuracy,
                "review_rate": float(review.mean()),
                "review_bucket_error_rate": review_error_rate,
            }
            key = (score, accepted_accuracy, coverage)
            best_key = None if best is None else (
                best["correctness_f1"],
                best["accepted_accuracy"],
                best["coverage"],
            )
            if best is None or key > best_key:
                best = candidate
    if best is None:
        raise RuntimeError("Unable to select a review policy")
    return best


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--specialist-validation",
        type=Path,
        default=ROOT / "outputs/lite_v3/final_baseline_specialist.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/lite_v3/calibration_specialist.json",
    )
    args = parser.parse_args()

    specialist_validation = json.loads(args.specialist_validation.read_text(encoding="utf-8"))["specialist"]
    pair_set = {
        frozenset(map(str, pair)) for pair in specialist_validation["deployment_pairs"]
    }
    specialist_threshold = float(specialist_validation["deployment_threshold"])

    rows = load_v5_rows()
    labels = labels_for_view(load_json(DEFAULT_CONTRACT), "full43")
    label_index = {label: index for index, label in enumerate(labels)}
    all_scores: list[np.ndarray] = []
    all_specialist_scores: list[np.ndarray] = []
    all_truth: list[int] = []

    started = time.perf_counter()
    for fold_number, (repeat, fold, train_rows, validation_rows) in enumerate(load_folds(rows), 1):
        print(f"calibration fold {fold_number}/12 repeat={repeat} fold={fold}", flush=True)
        y_train = [str(row["category"]) for row in train_rows]
        y_validation = [str(row["category"]) for row in validation_rows]

        x_train, x_validation = matrices(train_rows, validation_rows, BASE_FEATURE)
        base = classifier(BASE_CLASSIFIER).fit(x_train, y_train)
        all_scores.append(_aligned_scores(base, x_validation, labels))

        s_train, s_validation = matrices(train_rows, validation_rows, SPECIALIST_FEATURE)
        specialist = classifier(SPECIALIST_CLASSIFIER).fit(s_train, y_train)
        all_specialist_scores.append(_aligned_scores(specialist, s_validation, labels))
        all_truth.extend(label_index[label] for label in y_validation)

    scores = np.vstack(all_scores)
    specialist_scores = np.vstack(all_specialist_scores)
    truth = np.asarray(all_truth, dtype=int)

    def objective(log_temperature: float) -> float:
        temperature = math.exp(float(log_temperature))
        probabilities = _softmax(scores, temperature)
        probabilities = _apply_specialist(
            probabilities,
            specialist_scores,
            labels,
            pair_set,
            specialist_threshold,
        )
        return float(log_loss(truth, probabilities, labels=np.arange(len(labels))))

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
    policy = _policy(probabilities, truth)
    top1 = np.argmax(probabilities, axis=1)

    payload = {
        "validation": "12 frozen grouped DEVELOPMENT folds; specialist trained inside each outer TRAIN fold",
        "lockbox_accessed": False,
        "rows_oof": int(len(truth)),
        "temperature": round(float(temperature), 6),
        "nll": round(float(objective(math.log(temperature))), 6),
        "top1_accuracy": round(float((top1 == truth).mean()), 6),
        "threshold": policy["threshold"],
        "margin_threshold": policy["margin_threshold"],
        "policy_objective": "maximize F1 for OOF top1-correctness detection; tie-break accepted accuracy then coverage",
        "correctness_f1": round(float(policy["correctness_f1"]), 6),
        "coverage": round(float(policy["coverage"]), 6),
        "accepted_accuracy": round(float(policy["accepted_accuracy"]), 6),
        "review_rate": round(float(policy["review_rate"]), 6),
        "review_bucket_error_rate": round(float(policy["review_bucket_error_rate"]), 6),
        "specialist_pairs": [list(pair) for pair in specialist_validation["deployment_pairs"]],
        "specialist_threshold": specialist_threshold,
        "quality_note": (
            "Use cross-repeat metrics from final_baseline_specialist.json for the honest quality claim; "
            "this file fits deployment confidence/calibration on DEVELOPMENT OOF predictions."
        ),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
