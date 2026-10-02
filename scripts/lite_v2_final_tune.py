from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "backend"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from scripts.lite_v2_feature_tune import Recipe, make_matrices  # noqa: E402
from scripts.lite_v2_sprint import (  # noqa: E402
    ClassifierRecipe,
    aligned_probabilities,
    classifier,
    load_folds,
)
from scripts.v5_dataset import load_v5_rows  # noqa: E402
from scripts.v5_experiment import DEFAULT_CONTRACT, labels_for_view, load_json, operator_metrics  # noqa: E402


FEATURE = Recipe("word11", (1, 1), (3, 5))
CS = (0.15, 0.20, 0.25, 0.35, 0.50, 0.65, 0.80, 1.0, 1.2, 1.5, 2.0, 3.0)


def mean_metrics(records):
    keys = ("top1_accuracy", "top3_accuracy", "macro_f1", "true_label_mrr")
    return {key: round(mean(float(item[key]) for item in records), 9) for key in keys}


def quality_key(metrics):
    return (metrics["top1_accuracy"], metrics["macro_f1"], metrics["top3_accuracy"], metrics["true_label_mrr"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/lite_v2/final_tune.json")
    args = parser.parse_args()

    rows = load_v5_rows()
    labels = labels_for_view(load_json(DEFAULT_CONTRACT), "full43")
    candidates = [
        ClassifierRecipe(f"svc_c{c}_{'bal' if weight else 'none'}", "svc", c, weight)
        for c in CS
        for weight in ("balanced", None)
    ]
    scores = {candidate.name: [] for candidate in candidates}
    timings = {candidate.name: 0.0 for candidate in candidates}
    started = time.perf_counter()

    for fold_index, (_repeat, _fold, train_rows, validation_rows) in enumerate(load_folds(rows), 1):
        print(f"fold {fold_index}/12", flush=True)
        y_train = [str(row["category"]) for row in train_rows]
        y_validation = [str(row["category"]) for row in validation_rows]
        x_train, x_validation = make_matrices(train_rows, validation_rows, FEATURE)
        for candidate in candidates:
            step = time.perf_counter()
            model = classifier(candidate)
            model.fit(x_train, y_train)
            probabilities = aligned_probabilities(model, x_validation, labels, "svc")
            timings[candidate.name] += time.perf_counter() - step
            scores[candidate.name].append(operator_metrics(y_validation, probabilities, labels))

    results = []
    for candidate in candidates:
        results.append(
            {
                "classifier": candidate.__dict__,
                "metrics": mean_metrics(scores[candidate.name]),
                "elapsed_seconds": round(timings[candidate.name], 3),
            }
        )
    results.sort(key=lambda item: quality_key(item["metrics"]), reverse=True)
    payload = {
        "feature": FEATURE.__dict__,
        "metadata_scale": 0.35,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "winner": results[0],
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"winner": results[0], "top6": results[:6], "elapsed_seconds": payload["elapsed_seconds"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
