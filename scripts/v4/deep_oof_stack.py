from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

from app.ml.v4.evaluation import classification_metrics  # noqa: E402
from app.ml.v4.stacking import assert_oof_meta_split  # noqa: E402


def _load(view: str) -> tuple[dict[str, dict[tuple[int, str], str]], dict[tuple[int, str], str]]:
    models: dict[str, dict[tuple[int, str], str]] = {}
    truth: dict[tuple[int, str], str] = {}
    for path in sorted((PROJECT_ROOT / "artifacts/gpu_research_v4/deep").glob(f"*-{view}.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("status") != "MEASURED_DEEP_OPTIMIZATION":
            continue
        rows = payload.get("predictions", [])
        if not rows:
            continue
        name = str(payload["candidate_id"])
        models[name] = {}
        for row in rows:
            key = (int(row["repeat"]), str(row["request_id"]))
            models[name][key] = str(row["prediction"])
            truth[key] = str(row["truth"])
    if len(models) < 3:
        raise RuntimeError(f"need at least three deep finalists, found {sorted(models)}")
    return models, truth


def _objective(metrics: dict[str, Any], view: str) -> float:
    if view == "top15":
        return metrics["macro_f1"] + 0.25 * metrics["accuracy"] + 0.15 * metrics["worst_class_f1"]
    return metrics["macro_f1"] + 0.25 * metrics["balanced_accuracy"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--view", choices=("top15", "full43"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    pointer = json.loads((PROJECT_ROOT / "artifacts/gpu_research_v4/protocol.json").read_text())
    protocol = json.loads((PROJECT_ROOT / pointer["protocol_path"]).read_text())
    models, truth = _load(args.view)
    names = sorted(models)
    common = set.intersection(*(set(values) for values in models.values()))

    tuning = []
    tune_fold = next(fold for fold in protocol["folds"] if fold["repeat"] == 0 and fold["fold"] == 0)
    tune_train = [item for item in tune_fold["train_request_ids"] if (0, item) in common]
    tune_validation = [item for item in tune_fold["validation_request_ids"] if (0, item) in common]
    assert_oof_meta_split(tune_train, tune_validation, [item for repeat, item in common if repeat == 0])
    for c in (0.1, 0.3, 1.0, 3.0, 10.0):
        encoder = OneHotEncoder(handle_unknown="ignore")
        x_train = encoder.fit_transform([[models[name][(0, item)] for name in names] for item in tune_train])
        x_validation = encoder.transform([[models[name][(0, item)] for name in names] for item in tune_validation])
        meta = LogisticRegression(C=c, max_iter=2000, class_weight="balanced", random_state=20260917)
        meta.fit(x_train, [truth[(0, item)] for item in tune_train])
        predicted = [str(value) for value in meta.predict(x_validation)]
        labels = sorted({truth[(0, item)] for item in [*tune_train, *tune_validation]})
        metrics = classification_metrics([truth[(0, item)] for item in tune_validation], predicted, labels=labels)
        tuning.append({"C": c, "objective": round(_objective(metrics, args.view), 8), "metrics": metrics})
    selected = max(tuning, key=lambda row: row["objective"])

    repeat_metrics = {}
    majority_metrics = {}
    predictions = []
    for repeat in (0, 1, 2):
        stacked: dict[str, str] = {}
        majority: dict[str, str] = {}
        for fold in protocol["folds"]:
            if int(fold["repeat"]) != repeat:
                continue
            train = [item for item in fold["train_request_ids"] if (repeat, item) in common]
            validation = [item for item in fold["validation_request_ids"] if (repeat, item) in common]
            assert_oof_meta_split(train, validation, [item for value, item in common if value == repeat])
            encoder = OneHotEncoder(handle_unknown="ignore")
            x_train = encoder.fit_transform([[models[name][(repeat, item)] for name in names] for item in train])
            x_validation = encoder.transform(
                [[models[name][(repeat, item)] for name in names] for item in validation]
            )
            meta = LogisticRegression(
                C=float(selected["C"]), max_iter=2000, class_weight="balanced", random_state=20260917 + repeat
            )
            meta.fit(x_train, [truth[(repeat, item)] for item in train])
            for item, value in zip(validation, meta.predict(x_validation), strict=True):
                stacked[item] = str(value)
                votes = Counter(models[name][(repeat, item)] for name in names)
                majority[item] = sorted(votes, key=lambda label: (-votes[label], label))[0]
        evaluated = sorted(stacked)
        labels = sorted({truth[(repeat, item)] for item in evaluated})
        repeat_metrics[repeat] = classification_metrics(
            [truth[(repeat, item)] for item in evaluated], [stacked[item] for item in evaluated], labels=labels
        )
        majority_metrics[repeat] = classification_metrics(
            [truth[(repeat, item)] for item in evaluated], [majority[item] for item in evaluated], labels=labels
        )
        predictions.extend(
            {
                "repeat": repeat,
                "request_id": item,
                "truth": truth[(repeat, item)],
                "prediction": stacked[item],
                "majority": majority[item],
            }
            for item in evaluated
        )
    summary = defaultdict(list)
    for metrics in repeat_metrics.values():
        for key in ("accuracy", "macro_f1", "weighted_f1", "balanced_accuracy", "worst_class_f1"):
            summary[key].append(metrics[key])
    stability = {
        key: {"mean": round(float(np.mean(values)), 6), "std": round(float(np.std(values)), 6)}
        for key, values in summary.items()
    }
    payload = {
        "candidate_id": f"category/deep-oof-stack/{args.view}/v4",
        "status": "MEASURED_DEEP_STRICT_OOF",
        "view": args.view,
        "dataset_sha256": protocol["dataset_sha256"],
        "split_sha256": protocol["split_sha256"],
        "sealed_holdout_accessed": False,
        "base_models": names,
        "selected": selected,
        "repeat_metrics": repeat_metrics,
        "majority_repeat_metrics": majority_metrics,
        "stability": stability,
        "tuning": tuning,
        "predictions": predictions,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "selected": selected, "stability": stability}, ensure_ascii=False))


if __name__ == "__main__":
    main()
