from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from statistics import median
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "backend"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from scripts.v5_2_qwen4b_lite_sprint import (  # noqa: E402
    FoldStore,
    Recipe,
    _apply_specialist,
    _discover_pairs,
    _evaluate_recipe,
    _mean_metrics,
    _quality_key,
    _specialist_fold,
)
from scripts.v5_dataset import DEFAULT_DATASET, load_v5_rows  # noqa: E402
from scripts.v5_experiment import DEFAULT_CONTRACT, DEFAULT_PROTOCOL, load_json, operator_metrics  # noqa: E402

WINNER = Recipe(
    instruction="posttech_tight_c",
    c=1.0,
    blend="svc90_proto05_knn05",
    metadata_scale=0.75,
    knn_neighbors=11,
    prototype_temperature=0.08,
)


def _score_payloads(
    payloads: Sequence[Mapping[str, Any]],
    pairs: Sequence[tuple[str, str]],
    threshold: float,
) -> dict[str, Any]:
    pair_set = {frozenset(pair) for pair in pairs}
    fold_metrics = []
    corrected = introduced = 0
    for payload in payloads:
        probabilities, fixed, broken = _apply_specialist(payload, pair_set, threshold)
        fold_metrics.append(operator_metrics(payload["truth"], probabilities, payload["labels"]))
        corrected += fixed
        introduced += broken
    return {
        "metrics": _mean_metrics(fold_metrics),
        "corrected_errors": corrected,
        "introduced_errors": introduced,
    }


def _select_threshold(
    payloads: Sequence[Mapping[str, Any]],
    pairs: Sequence[tuple[str, str]],
) -> dict[str, Any]:
    candidates = []
    for threshold in (0.0, 0.05, 0.10, 0.15, 0.20, 0.30):
        scored = _score_payloads(payloads, pairs, threshold)
        candidates.append({"threshold": threshold, **scored})
    candidates.sort(key=lambda item: _quality_key(item["metrics"]), reverse=True)
    return candidates[0]


def validate_view(base_eval: Mapping[str, Any]) -> dict[str, Any]:
    raw = base_eval["predictions"] or []
    specialist_folds = [_specialist_fold(item) for item in raw]
    selected_by_split: list[list[tuple[str, str]]] = []
    threshold_by_split: list[float] = []
    heldout_fold_metrics = []
    corrected = introduced = 0
    splits = []

    for heldout_repeat in (0, 1, 2):
        train_payloads = [item for item in specialist_folds if int(item["repeat"]) != heldout_repeat]
        test_payloads = [item for item in specialist_folds if int(item["repeat"]) == heldout_repeat]
        pairs = _discover_pairs(train_payloads)
        selected_by_split.append(pairs)
        tuned = _select_threshold(train_payloads, pairs) if pairs else {
            "threshold": 0.0,
            "metrics": base_eval["metrics"],
            "corrected_errors": 0,
            "introduced_errors": 0,
        }
        threshold = float(tuned["threshold"])
        threshold_by_split.append(threshold)
        pair_set = {frozenset(pair) for pair in pairs}
        split_metrics = []
        split_corrected = split_introduced = 0
        for payload in test_payloads:
            probabilities, fixed, broken = _apply_specialist(payload, pair_set, threshold)
            metrics = operator_metrics(payload["truth"], probabilities, payload["labels"])
            heldout_fold_metrics.append(metrics)
            split_metrics.append(metrics)
            corrected += fixed
            introduced += broken
            split_corrected += fixed
            split_introduced += broken
        splits.append(
            {
                "heldout_repeat": heldout_repeat,
                "pairs": pairs,
                "threshold": threshold,
                "train_metrics": tuned["metrics"],
                "heldout_metrics": _mean_metrics(split_metrics),
                "heldout_corrected_errors": split_corrected,
                "heldout_introduced_errors": split_introduced,
            }
        )

    cross_metrics = _mean_metrics(heldout_fold_metrics)
    pair_counts: Counter[tuple[str, str]] = Counter()
    for pairs in selected_by_split:
        pair_counts.update(tuple(pair) for pair in pairs)
    deployment_pairs = sorted(pair for pair, count in pair_counts.items() if count >= 2)
    deployment_threshold = float(median(threshold_by_split)) if threshold_by_split else 0.0
    deployment_score = _score_payloads(specialist_folds, deployment_pairs, deployment_threshold)
    base = base_eval["metrics"]
    enabled = bool(deployment_pairs) and _quality_key(cross_metrics) > _quality_key(base)
    return {
        "enabled": enabled,
        "validation": "leave-one-repeat-out specialist selection; each held-out repeat is never used to select its pairs or threshold",
        "base_metrics": dict(base),
        "cross_repeat_metrics": cross_metrics if enabled else dict(base),
        "cross_repeat_corrected_errors": corrected if enabled else 0,
        "cross_repeat_introduced_errors": introduced if enabled else 0,
        "cross_repeat_top1_delta": round(cross_metrics["top1_accuracy"] - base["top1_accuracy"], 9) if enabled else 0.0,
        "cross_repeat_macro_f1_delta": round(cross_metrics["macro_f1"] - base["macro_f1"], 9) if enabled else 0.0,
        "cross_repeat_top3_delta": round(cross_metrics["top3_accuracy"] - base["top3_accuracy"], 9) if enabled else 0.0,
        "deployment_pairs": deployment_pairs if enabled else [],
        "deployment_threshold": deployment_threshold if enabled else None,
        "deployment_fit_metrics": deployment_score["metrics"] if enabled else dict(base),
        "deployment_fit_corrected_errors": deployment_score["corrected_errors"] if enabled else 0,
        "deployment_fit_introduced_errors": deployment_score["introduced_errors"] if enabled else 0,
        "pair_selection_frequency": [
            {"pair": list(pair), "selected_splits": count}
            for pair, count in sorted(pair_counts.items(), key=lambda item: (-item[1], item[0]))
        ],
        "splits": splits,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Leakage-safe cross-repeat validation for the Qwen4B Lite TF-IDF specialist")
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=ROOT / "artifacts/gpu_research_v5/embedding_cache",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/qwen4b_lite/specialist_validation.json",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    store = FoldStore(
        rows=load_v5_rows(DEFAULT_DATASET),
        protocol=load_json(DEFAULT_PROTOCOL),
        contract=load_json(DEFAULT_CONTRACT),
        cache_root=args.cache_root,
        batch_size=args.batch_size,
    )
    folds = [(repeat, fold) for repeat in range(3) for fold in range(4)]
    top15 = _evaluate_recipe(store, "top15", WINNER, folds, keep_predictions=True)
    full43 = _evaluate_recipe(store, "full43", WINNER, folds, keep_predictions=True)
    payload = {
        "status": "complete",
        "winner": WINNER.as_dict(),
        "lockbox_accessed": False,
        "top15": validate_view(top15),
        "full43": validate_view(full43),
        "embedding_cache_hits": store.embedder.hits,
        "embedding_cache_misses": store.embedder.misses,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
