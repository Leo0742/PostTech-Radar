from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

from app.ml.v4.evaluation import classification_metrics  # noqa: E402
from app.ml.v4.stacking import assert_oof_meta_split  # noqa: E402


def _protocol() -> dict[str, Any]:
    pointer = json.loads((PROJECT_ROOT / "artifacts/gpu_research_v4/protocol.json").read_text())
    return json.loads((PROJECT_ROOT / pointer["protocol_path"]).read_text())


def _load_predictions(view: str) -> tuple[list[str], dict[str, str], dict[str, dict[str, str]], list[str]]:
    paths = sorted((PROJECT_ROOT / "artifacts/gpu_research_v4/embeddings").glob(f"*-{view}.json"))
    truth: dict[str, str] = {}
    models: dict[str, dict[str, str]] = {}
    sources: list[str] = []
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("status") != "MEASURED_GPU_OOF":
            continue
        rows = payload["predictions"].get("embedding_metadata_lr", [])
        if not rows:
            continue
        name = payload["candidate_id"]
        models[name] = {str(row["request_id"]): str(row["prediction"]) for row in rows}
        for row in rows:
            request_id = str(row["request_id"])
            label = str(row["truth"])
            if request_id in truth and truth[request_id] != label:
                raise ValueError(f"truth mismatch for {request_id}")
            truth[request_id] = label
        sources.append(str(path.relative_to(PROJECT_ROOT)))
    if len(models) < 2:
        raise RuntimeError("at least two measured OOF models are required")
    common = sorted(set.intersection(*(set(rows) for rows in models.values())))
    return common, truth, models, sources


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--view", choices=("top15", "full43"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    protocol = _protocol()
    request_ids, truth, models, sources = _load_predictions(args.view)
    model_names = sorted(models)
    base_ids = set(request_ids)
    stacked: dict[str, str] = {}
    majority: dict[str, str] = {}
    fold_audit = []
    for fold in protocol["folds"]:
        if int(fold["repeat"]) != 0:
            continue
        train_ids = [item for item in fold["train_request_ids"] if item in base_ids]
        validation_ids = [item for item in fold["validation_request_ids"] if item in base_ids]
        assert_oof_meta_split(train_ids, validation_ids, request_ids)
        encoder = OneHotEncoder(handle_unknown="ignore")
        x_train_raw = [[models[name][item] for name in model_names] for item in train_ids]
        x_validation_raw = [[models[name][item] for name in model_names] for item in validation_ids]
        x_train = encoder.fit_transform(x_train_raw)
        x_validation = encoder.transform(x_validation_raw)
        meta = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=int(fold["seed"]))
        meta.fit(x_train, [truth[item] for item in train_ids])
        for item, value in zip(validation_ids, meta.predict(x_validation), strict=True):
            stacked[item] = str(value)
        for item in validation_ids:
            votes = Counter(models[name][item] for name in model_names)
            majority[item] = sorted(votes, key=lambda value: (-votes[value], value))[0]
        fold_audit.append({"fold": fold["fold"], "train": len(train_ids), "validation": len(validation_ids)})

    evaluated = sorted(set(stacked) & set(majority))
    labels = sorted(set(truth[item] for item in evaluated))
    payload = {
        "candidate_id": f"category/oof-stack/{args.view}/v4",
        "status": "MEASURED_STRICT_OOF",
        "view": args.view,
        "dataset_sha256": protocol["dataset_sha256"],
        "split_sha256": protocol["split_sha256"],
        "sealed_holdout_accessed": False,
        "real_only_evaluation": True,
        "base_models": model_names,
        "sources": sources,
        "fold_audit": fold_audit,
        "metrics": {
            "oof_logistic_stack": classification_metrics(
                [truth[item] for item in evaluated], [stacked[item] for item in evaluated], labels=labels
            ),
            "majority_vote": classification_metrics(
                [truth[item] for item in evaluated], [majority[item] for item in evaluated], labels=labels
            ),
        },
        "predictions": [
            {"request_id": item, "truth": truth[item], "stack": stacked[item], "majority": majority[item]}
            for item in evaluated
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "metrics": payload["metrics"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
