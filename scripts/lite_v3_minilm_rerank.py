from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np
from sklearn.svm import LinearSVC

ROOT = Path(__file__).resolve().parents[1]
for import_path in (ROOT, ROOT / "backend"):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from scripts.lite_v2_feature_tune import Recipe as TextRecipe  # noqa: E402
from scripts.lite_v2_feature_tune import make_matrices  # noqa: E402
from scripts.lite_v2_sprint import load_folds  # noqa: E402
from scripts.lite_v3_local_max import ensure_minilm_embeddings  # noqa: E402
from scripts.v5_dataset import load_v5_rows  # noqa: E402
from scripts.v5_experiment import (  # noqa: E402
    DEFAULT_CONTRACT,
    labels_for_view,
    load_json,
    operator_metrics,
)

SEED = 20260920


def _softmax(scores: np.ndarray, temperature: float) -> np.ndarray:
    scaled = np.asarray(scores, dtype=float) / max(float(temperature), 1e-6)
    scaled -= scaled.max(axis=1, keepdims=True)
    exp = np.exp(scaled)
    return exp / exp.sum(axis=1, keepdims=True)


def _align_scores(model: Any, matrix: Any, labels: list[str]) -> np.ndarray:
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


def _mean_metrics(records: list[dict[str, Any]]) -> dict[str, float]:
    keys = ("top1_accuracy", "top3_accuracy", "macro_f1", "true_label_mrr")
    return {key: round(mean(float(record[key]) for record in records), 9) for key in keys}


def _quality_key(metrics: dict[str, Any]) -> tuple[float, float, float, float]:
    return (
        float(metrics["top1_accuracy"]),
        float(metrics["macro_f1"]),
        float(metrics["top3_accuracy"]),
        float(metrics["true_label_mrr"]),
    )


def _rerank(
    base_probabilities: np.ndarray,
    semantic_scores: np.ndarray,
    *,
    gate_margin: float,
    semantic_margin: float,
) -> tuple[np.ndarray, int]:
    result = np.asarray(base_probabilities, dtype=float).copy()
    triggered = 0
    for row_index in range(len(result)):
        order = np.argsort(-result[row_index])
        first, second = int(order[0]), int(order[1])
        base_margin = float(result[row_index, first] - result[row_index, second])
        if base_margin > gate_margin:
            continue
        triggered += 1
        margin = float(semantic_scores[row_index, second] - semantic_scores[row_index, first])
        if margin > semantic_margin:
            result[row_index, first], result[row_index, second] = (
                result[row_index, second],
                result[row_index, first],
            )
    return result, triggered


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, default=ROOT / "outputs/lite_v3/minilm_256.npy")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-seq-length", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.166829)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/lite_v3/minilm_rerank.json")
    args = parser.parse_args()

    rows = load_v5_rows()
    labels = labels_for_view(load_json(DEFAULT_CONTRACT), "full43")
    embeddings = ensure_minilm_embeddings(
        rows,
        args.cache,
        device=args.device,
        batch_size=args.batch_size,
        max_seq_length=args.max_seq_length,
    )
    row_index = {str(row["request_id"]): index for index, row in enumerate(rows)}
    text_recipe = TextRecipe("word11", (1, 1), (3, 5))
    semantic_cs = (0.2, 0.5, 1.0, 2.0)
    gate_margins = (0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40)
    semantic_margins = (0.0, 0.05, 0.10, 0.20, 0.30)
    folds_by_c: dict[float, list[dict[str, Any]]] = {c: [] for c in semantic_cs}
    base_metrics: list[dict[str, Any]] = []

    started = time.perf_counter()
    for fold_number, (repeat, fold, train_rows, validation_rows) in enumerate(load_folds(rows), 1):
        print(f"rerank fold {fold_number}/12 repeat={repeat} fold={fold}", flush=True)
        y_train = [str(row["category"]) for row in train_rows]
        y_validation = [str(row["category"]) for row in validation_rows]
        train_indices = [row_index[str(row["request_id"])] for row in train_rows]
        validation_indices = [row_index[str(row["request_id"])] for row in validation_rows]
        e_train = embeddings[train_indices]
        e_validation = embeddings[validation_indices]

        x_train, x_validation = make_matrices(train_rows, validation_rows, text_recipe)
        base_model = LinearSVC(
            C=0.35,
            class_weight="balanced",
            max_iter=10000,
            random_state=SEED + repeat * 10 + fold,
        ).fit(x_train, y_train)
        base_probabilities = _softmax(_align_scores(base_model, x_validation, labels), args.temperature)
        base_metrics.append(operator_metrics(y_validation, base_probabilities, labels))

        for c in semantic_cs:
            semantic_model = LinearSVC(
                C=c,
                class_weight="balanced",
                max_iter=10000,
                random_state=SEED + repeat * 10 + fold,
            ).fit(e_train, y_train)
            folds_by_c[c].append(
                {
                    "repeat": repeat,
                    "fold": fold,
                    "truth": y_validation,
                    "base": base_probabilities,
                    "semantic_scores": _align_scores(semantic_model, e_validation, labels),
                }
            )

    candidates = [
        (c, gate_margin, semantic_margin)
        for c in semantic_cs
        for gate_margin in gate_margins
        for semantic_margin in semantic_margins
    ]

    def evaluate(payloads: list[dict[str, Any]], candidate: tuple[float, float, float]) -> dict[str, Any]:
        c, gate_margin, semantic_margin = candidate
        records = []
        triggered = rows_seen = 0
        for payload in payloads:
            probabilities, local_triggered = _rerank(
                payload["base"],
                payload["semantic_scores"],
                gate_margin=gate_margin,
                semantic_margin=semantic_margin,
            )
            records.append(operator_metrics(payload["truth"], probabilities, labels))
            triggered += local_triggered
            rows_seen += len(payload["truth"])
        return {
            "candidate": {
                "semantic_c": c,
                "gate_margin": gate_margin,
                "semantic_margin": semantic_margin,
            },
            "metrics": _mean_metrics(records),
            "semantic_run_rate": round(triggered / max(rows_seen, 1), 6),
        }

    all_payloads_by_c = folds_by_c
    global_results = []
    for candidate in candidates:
        c = candidate[0]
        global_results.append(evaluate(all_payloads_by_c[c], candidate))
    global_results.sort(key=lambda item: _quality_key(item["metrics"]), reverse=True)

    heldout_metrics = []
    split_results = []
    selected = []
    for heldout_repeat in (0, 1, 2):
        train_scores = []
        for candidate in candidates:
            c = candidate[0]
            payloads = [item for item in all_payloads_by_c[c] if int(item["repeat"]) != heldout_repeat]
            train_scores.append(evaluate(payloads, candidate))
        train_scores.sort(key=lambda item: _quality_key(item["metrics"]), reverse=True)
        winner = train_scores[0]
        selected_candidate = (
            float(winner["candidate"]["semantic_c"]),
            float(winner["candidate"]["gate_margin"]),
            float(winner["candidate"]["semantic_margin"]),
        )
        selected.append(selected_candidate)
        test_payloads = [
            item
            for item in all_payloads_by_c[selected_candidate[0]]
            if int(item["repeat"]) == heldout_repeat
        ]
        test_score = evaluate(test_payloads, selected_candidate)
        heldout_metrics.append(test_score["metrics"])
        split_results.append(
            {
                "heldout_repeat": heldout_repeat,
                "selected": winner,
                "heldout": test_score,
            }
        )

    selection_counts = Counter(selected)
    payload = {
        "stage": "minilm_conditional_top2_rerank",
        "validation": "leave-one-repeat-out rule selection over the 12 frozen grouped DEVELOPMENT folds",
        "lockbox_accessed": False,
        "baseline": _mean_metrics(base_metrics),
        "cross_repeat_metrics": _mean_metrics(heldout_metrics),
        "global_development_fit": global_results[0],
        "selection_frequency": [
            {
                "candidate": {
                    "semantic_c": item[0][0],
                    "gate_margin": item[0][1],
                    "semantic_margin": item[0][2],
                },
                "selected_splits": item[1],
            }
            for item in selection_counts.most_common()
        ],
        "splits": split_results,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
