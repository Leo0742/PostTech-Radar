from __future__ import annotations

import argparse
import itertools
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.v5_experiment import candidate_quality_key, operator_metrics


DEFAULT_RESULTS = (
    ROOT
    / "artifacts"
    / "gpu_research_v5"
    / "runs"
    / "qwen8b_quality_sprint"
    / "quality_head_refine"
)


def load_results(root: Path) -> list[dict[str, Any]]:
    values: list[dict[str, Any]] = []
    for path in sorted(Path(root).glob("*/result.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if payload.get("status") != "complete" or not payload.get("predictions"):
            continue
        values.append(payload)
    return values


def fold_key(result: dict[str, Any]) -> tuple[str, int, int]:
    candidate = result["candidate"]
    return str(candidate["view"]), int(candidate["repeat"]), int(candidate["fold"])


def probability_map(result: dict[str, Any]) -> tuple[list[str], dict[str, tuple[str, np.ndarray]]]:
    labels = [str(value) for value in result["prediction_labels"]]
    rows: dict[str, tuple[str, np.ndarray]] = {}
    for item in result["predictions"]:
        rows[str(item["request_id"])] = (
            str(item["truth"]),
            np.asarray(item["probabilities"], dtype=float),
        )
    return labels, rows


def evaluate_blend(
    first: dict[str, Any],
    second: dict[str, Any],
    *,
    first_weight: float,
) -> dict[str, Any]:
    labels_a, rows_a = probability_map(first)
    labels_b, rows_b = probability_map(second)
    if labels_a != labels_b:
        raise ValueError("Prediction label order differs between ensemble members")
    ids = sorted(set(rows_a).intersection(rows_b))
    if len(ids) != len(rows_a) or len(ids) != len(rows_b):
        raise ValueError("Validation request IDs differ between ensemble members")
    truth: list[str] = []
    matrix: list[np.ndarray] = []
    for request_id in ids:
        truth_a, probs_a = rows_a[request_id]
        truth_b, probs_b = rows_b[request_id]
        if truth_a != truth_b:
            raise ValueError(f"Truth mismatch for request {request_id}")
        truth.append(truth_a)
        mixed = first_weight * probs_a + (1.0 - first_weight) * probs_b
        mixed /= mixed.sum()
        matrix.append(mixed)
    return operator_metrics(truth, np.asarray(matrix, dtype=float), labels_a)


def aggregate(values: list[dict[str, Any]]) -> dict[str, float]:
    names = ("top1_accuracy", "macro_f1", "top3_accuracy", "true_label_mrr")
    return {
        name: round(statistics.fmean(float(item[name]) for item in values), 6)
        for name in names
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze leakage-safe probability ensembles across matched V5 folds")
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--weights", default="0.25,0.5,0.75")
    parser.add_argument("--top", type=int, default=20)
    args = parser.parse_args()

    weights = [float(value) for value in str(args.weights).split(",") if value.strip()]
    results = load_results(args.results_root)
    by_fold: dict[tuple[str, int, int], dict[str, dict[str, Any]]] = defaultdict(dict)
    for result in results:
        head = str(result["candidate"]["head"])
        by_fold[fold_key(result)][head] = result

    heads = sorted({str(result["candidate"]["head"]) for result in results})
    rows: list[dict[str, Any]] = []
    for first_head, second_head in itertools.combinations(heads, 2):
        for weight in weights:
            fold_metrics: list[dict[str, Any]] = []
            for fold, members in by_fold.items():
                if first_head not in members or second_head not in members:
                    continue
                fold_metrics.append(
                    evaluate_blend(
                        members[first_head],
                        members[second_head],
                        first_weight=weight,
                    )
                )
            if not fold_metrics:
                continue
            metrics = aggregate(fold_metrics)
            rows.append(
                {
                    "first_head": first_head,
                    "second_head": second_head,
                    "first_weight": weight,
                    "second_weight": round(1.0 - weight, 6),
                    "folds": len(fold_metrics),
                    "metrics": metrics,
                }
            )

    rows.sort(key=lambda item: candidate_quality_key(item["metrics"]), reverse=True)
    print(json.dumps(rows[: max(1, args.top)], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
