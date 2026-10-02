from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from importlib.metadata import version
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

from app.core.config import DATABASE_PATH  # noqa: E402
from app.ml.v4.evaluation import classification_metrics  # noqa: E402
from app.services.data_service import load_rows  # noqa: E402


def _sample(row: dict[str, object], label_to_id: dict[str, int]) -> dict[str, object]:
    return {
        "utterance": str(row["description"]),
        "label": label_to_id[str(row["category"])],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Screen DeepPavlov AutoIntent on a frozen v4 fold")
    parser.add_argument("--view", choices=("top15", "full43"), default="top15")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260917)
    parser.add_argument("--preset", default="classic-light")
    args = parser.parse_args()

    from autointent import Dataset, Pipeline

    pointer = json.loads((PROJECT_ROOT / "artifacts/gpu_research_v4/protocol.json").read_text(encoding="utf-8"))
    protocol = json.loads((PROJECT_ROOT / pointer["protocol_path"]).read_text(encoding="utf-8"))
    fold = next(
        item for item in protocol["folds"] if int(item["repeat"]) == 0 and int(item["fold"]) == args.fold
    )
    rows = [row for row in load_rows(DATABASE_PATH) if str(row.get("description") or "").strip()]
    counts = Counter(str(row["category"]) for row in rows)
    labels = sorted({label for label, _ in counts.most_common(15)} if args.view == "top15" else set(counts))
    label_to_id = {label: index for index, label in enumerate(labels)}
    by_id = {str(row["request_id"]): row for row in rows if str(row["category"]) in label_to_id}
    train_rows = [by_id[item] for item in fold["train_request_ids"] if item in by_id]
    validation_rows = [by_id[item] for item in fold["validation_request_ids"] if item in by_id]

    train_labels = {str(row["category"]) for row in train_rows}
    validation_labels = {str(row["category"]) for row in validation_rows}
    missing_train = sorted(set(labels) - train_labels)
    missing_validation = sorted(set(labels) - validation_labels)
    if missing_train or missing_validation:
        raise RuntimeError(
            "AutoIntent requires every class in its train and validation splits; "
            f"missing_train={missing_train}, missing_validation={missing_validation}"
        )

    mapping = {
        "train": [_sample(row, label_to_id) for row in train_rows],
        "validation": [_sample(row, label_to_id) for row in validation_rows],
        # Required by the package contract but never used to tune or score this run.
        "test": [_sample(row, label_to_id) for row in validation_rows],
        "intents": [{"id": index, "name": label} for index, label in enumerate(labels)],
    }
    dataset = Dataset.from_dict(mapping)
    pipeline = Pipeline.from_preset(args.preset, seed=args.seed)
    started = time.perf_counter()
    context = pipeline.fit(dataset, refit_after=False)
    train_seconds = time.perf_counter() - started
    infer_started = time.perf_counter()
    prediction_ids = pipeline.predict([str(row["description"]) for row in validation_rows])
    inference_seconds = time.perf_counter() - infer_started
    predicted = [labels[int(label)] for label in prediction_ids]
    truth = [str(row["category"]) for row in validation_rows]

    payload = {
        "candidate_id": f"category/deeppavlov-autointent-{args.preset}/{args.view}/v4",
        "status": "MEASURED_FROZEN_FOLD",
        "package": f"autointent=={version('autointent')}",
        "family": "DeepPavlov AutoIntent",
        "preset": args.preset,
        "preset_embedder": "intfloat/multilingual-e5-large-instruct",
        "view": args.view,
        "fold": args.fold,
        "seed": args.seed,
        "dataset_sha256": protocol["dataset_sha256"],
        "split_sha256": protocol["split_sha256"],
        "sealed_holdout_accessed": False,
        "real_only_train": True,
        "real_only_evaluation": True,
        "train_rows": len(train_rows),
        "validation_rows": len(validation_rows),
        "train_seconds": round(train_seconds, 3),
        "inference_seconds": round(inference_seconds, 3),
        "latency_ms_per_ticket": round(1000 * inference_seconds / len(validation_rows), 4),
        "context_repr": repr(context),
        "metrics": classification_metrics(truth, predicted, labels=labels),
        "predictions": [
            {
                "request_id": str(row["request_id"]),
                "truth": truth[index],
                "prediction": predicted[index],
            }
            for index, row in enumerate(validation_rows)
        ],
    }
    output = (
        PROJECT_ROOT
        / "artifacts/gpu_research_v4/coverage"
        / f"autointent-{args.preset}-{args.view}-f{args.fold}-s{args.seed}.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "metrics": payload["metrics"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
