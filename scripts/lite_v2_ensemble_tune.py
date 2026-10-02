from __future__ import annotations

import argparse
import itertools
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "backend"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from scripts.lite_v2_feature_tune import Recipe, make_matrices as tuned_matrices  # noqa: E402
from scripts.lite_v2_sprint import (  # noqa: E402
    ClassifierRecipe,
    FeatureRecipe,
    aligned_probabilities,
    classifier,
    load_folds,
    matrices as generic_matrices,
)
from scripts.v5_dataset import load_v5_rows  # noqa: E402
from scripts.v5_experiment import DEFAULT_CONTRACT, labels_for_view, load_json, operator_metrics  # noqa: E402


@dataclass(frozen=True)
class Candidate:
    name: str
    kind: str
    feature: dict[str, Any]
    classifier: ClassifierRecipe


CANDIDATES = [
    Candidate(
        "word11_c035_bal",
        "tuned",
        asdict(Recipe("word11", (1, 1), (3, 5))),
        ClassifierRecipe("svc_c0.35_bal", "svc", 0.35, "balanced"),
    ),
    Candidate(
        "word11_c065_bal",
        "tuned",
        asdict(Recipe("word11", (1, 1), (3, 5))),
        ClassifierRecipe("svc_c0.65_bal", "svc", 0.65, "balanced"),
    ),
    Candidate(
        "qwen90_c10_bal",
        "generic",
        asdict(FeatureRecipe("qwen90_struct035", "specialist", 30000, 60000, (1, 2), (3, 5), True, 0.35)),
        ClassifierRecipe("svc_c1.0_bal", "svc", 1.0, "balanced"),
    ),
    Candidate(
        "current48_c08_none",
        "generic",
        asdict(FeatureRecipe("current48_combined", "combined", 20000, 28000, (1, 3), (2, 6), False, 1.0)),
        ClassifierRecipe("svc_c0.8_none", "svc", 0.8, None),
    ),
    Candidate(
        "rich90_c08_none",
        "generic",
        asdict(FeatureRecipe("rich90_struct075", "combined", 30000, 60000, (1, 3), (3, 5), True, 0.75)),
        ClassifierRecipe("svc_c0.8_none", "svc", 0.8, None),
    ),
]


def _feature_object(candidate: Candidate):
    if candidate.kind == "tuned":
        return Recipe(**candidate.feature)
    return FeatureRecipe(**candidate.feature)


def _matrices(train_rows, validation_rows, candidate: Candidate):
    feature = _feature_object(candidate)
    if candidate.kind == "tuned":
        return tuned_matrices(train_rows, validation_rows, feature)
    return generic_matrices(train_rows, validation_rows, feature)


def _mean_metrics(records):
    keys = ("top1_accuracy", "top3_accuracy", "macro_f1", "true_label_mrr")
    return {key: round(mean(float(item[key]) for item in records), 9) for key in keys}


def _quality_key(metrics):
    return (metrics["top1_accuracy"], metrics["macro_f1"], metrics["top3_accuracy"], metrics["true_label_mrr"])


def _weight_grid(size: int, units: int = 5):
    for values in itertools.product(range(units + 1), repeat=size):
        if sum(values) != units:
            continue
        yield np.asarray(values, dtype=float) / float(units)


def _blend(fold: dict[str, Any], weights: np.ndarray) -> np.ndarray:
    stacked = np.stack(fold["probabilities"], axis=0)
    return np.tensordot(weights, stacked, axes=(0, 0))


def _score(folds: list[dict[str, Any]], weights: np.ndarray) -> dict[str, float]:
    records = [operator_metrics(fold["truth"], _blend(fold, weights), fold["labels"]) for fold in folds]
    return _mean_metrics(records)


def _select(train_folds: list[dict[str, Any]]) -> tuple[np.ndarray, dict[str, float]]:
    best_weights = None
    best_metrics = None
    for weights in _weight_grid(len(CANDIDATES)):
        metrics = _score(train_folds, weights)
        if best_metrics is None or _quality_key(metrics) > _quality_key(best_metrics):
            best_weights = weights
            best_metrics = metrics
    assert best_weights is not None and best_metrics is not None
    return best_weights, best_metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/lite_v2/ensemble.json")
    args = parser.parse_args()

    rows = load_v5_rows()
    labels = labels_for_view(load_json(DEFAULT_CONTRACT), "full43")
    fold_payloads = []

    for fold_index, (repeat, fold, train_rows, validation_rows) in enumerate(load_folds(rows), 1):
        print(f"fold {fold_index}/12 repeat={repeat} fold={fold}", flush=True)
        y_train = [str(row["category"]) for row in train_rows]
        y_validation = [str(row["category"]) for row in validation_rows]
        probabilities = []
        for candidate in CANDIDATES:
            x_train, x_validation = _matrices(train_rows, validation_rows, candidate)
            model = classifier(candidate.classifier)
            model.fit(x_train, y_train)
            probabilities.append(aligned_probabilities(model, x_validation, labels, "svc"))
        fold_payloads.append(
            {
                "repeat": repeat,
                "fold": fold,
                "truth": y_validation,
                "labels": labels,
                "probabilities": probabilities,
            }
        )

    individual = []
    for index, candidate in enumerate(CANDIDATES):
        weights = np.zeros(len(CANDIDATES), dtype=float)
        weights[index] = 1.0
        individual.append({"candidate": candidate.name, "metrics": _score(fold_payloads, weights)})

    heldout_records = []
    selected_weights = []
    splits = []
    for heldout_repeat in (0, 1, 2):
        train_folds = [fold for fold in fold_payloads if fold["repeat"] != heldout_repeat]
        test_folds = [fold for fold in fold_payloads if fold["repeat"] == heldout_repeat]
        weights, train_metrics = _select(train_folds)
        test_metrics = _score(test_folds, weights)
        selected_weights.append(weights)
        heldout_records.append(test_metrics)
        splits.append(
            {
                "heldout_repeat": heldout_repeat,
                "weights": {candidate.name: round(float(weight), 3) for candidate, weight in zip(CANDIDATES, weights)},
                "train_metrics": train_metrics,
                "heldout_metrics": test_metrics,
            }
        )

    stable = np.mean(np.stack(selected_weights, axis=0), axis=0)
    stable /= stable.sum()
    payload = {
        "validation": "leave-one-repeat-out ensemble weight selection",
        "candidates": [
            {
                "name": candidate.name,
                "kind": candidate.kind,
                "feature": candidate.feature,
                "classifier": asdict(candidate.classifier),
            }
            for candidate in CANDIDATES
        ],
        "individual_metrics": individual,
        "cross_repeat_metrics": _mean_metrics(heldout_records),
        "deployment_weights": {
            candidate.name: round(float(weight), 6) for candidate, weight in zip(CANDIDATES, stable)
        },
        "deployment_fit_metrics": _score(fold_payloads, stable),
        "splits": splits,
        "lockbox_accessed": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
