from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

from app.core.config import DATABASE_PATH  # noqa: E402
from app.ml.v4.evaluation import classification_metrics  # noqa: E402
from app.ml.v4.stacking import assert_oof_meta_split  # noqa: E402
from app.services.data_service import load_rows  # noqa: E402


def _rows(path: Path, field: str = "prediction") -> tuple[dict[str, str], dict[str, str]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    predictions = payload["predictions"]
    if predictions and "repeat" in predictions[0]:
        predictions = [row for row in predictions if int(row["repeat"]) == 0]
    return (
        {str(row["request_id"]): str(row[field]) for row in predictions},
        {str(row["request_id"]): str(row["truth"]) for row in predictions},
    )


def main() -> None:
    pointer = json.loads((PROJECT_ROOT / "artifacts/gpu_research_v4/protocol.json").read_text())
    protocol = json.loads((PROJECT_ROOT / pointer["protocol_path"]).read_text())
    structured, truth = _rows(PROJECT_ROOT / "artifacts/gpu_research_v4/deep/structured-full43.json")
    nli, nli_truth = _rows(PROJECT_ROOT / "artifacts/gpu_research_v4/nli/mdeberta-full43.json")
    base_models: dict[str, dict[str, str]] = {"structured": structured, "nli": nli}
    for path in sorted((PROJECT_ROOT / "artifacts/gpu_research_v4/embeddings").glob("*-full43.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows = payload["predictions"]["embedding_metadata_lr"]
        base_models[payload["candidate_id"]] = {
            str(row["request_id"]): str(row["prediction"]) for row in rows
        }
        for row in rows:
            if truth.get(str(row["request_id"]), str(row["truth"])) != str(row["truth"]):
                raise ValueError("truth mismatch")
    if truth != nli_truth:
        raise ValueError("NLI and structured OOF targets differ")
    common = sorted(set.intersection(*(set(values) for values in base_models.values())))
    names = sorted(base_models)
    stacked: dict[str, str] = {}
    for fold in protocol["folds"]:
        if int(fold["repeat"]) != 0:
            continue
        train = [item for item in fold["train_request_ids"] if item in common]
        validation = [item for item in fold["validation_request_ids"] if item in common]
        assert_oof_meta_split(train, validation, common)
        encoder = OneHotEncoder(handle_unknown="ignore")
        x_train = encoder.fit_transform([[base_models[name][item] for name in names] for item in train])
        x_validation = encoder.transform([[base_models[name][item] for name in names] for item in validation])
        meta = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=int(fold["seed"]))
        meta.fit(x_train, [truth[item] for item in train])
        for item, value in zip(validation, meta.predict(x_validation), strict=True):
            stacked[item] = str(value)
    evaluated = sorted(stacked)
    counts = Counter(str(row["category"]) for row in load_rows(DATABASE_PATH))
    labels = sorted(counts)
    metrics = classification_metrics(
        [truth[item] for item in evaluated], [stacked[item] for item in evaluated], labels=labels
    )
    baseline_metrics = classification_metrics(
        [truth[item] for item in evaluated], [structured[item] for item in evaluated], labels=labels
    )
    payload: dict[str, Any] = {
        "candidate_id": "category/rare-semantic-oof-stack/full43/v4",
        "status": "MEASURED_STRICT_OOF",
        "dataset_sha256": protocol["dataset_sha256"],
        "split_sha256": protocol["split_sha256"],
        "sealed_holdout_accessed": False,
        "real_only_evaluation": True,
        "base_models": names,
        "baseline_structured": baseline_metrics,
        "metrics": metrics,
        "predictions": [
            {"request_id": item, "truth": truth[item], "prediction": stacked[item]} for item in evaluated
        ],
    }
    destination = PROJECT_ROOT / "artifacts/gpu_research_v4/deep/rare-semantic-stack-full43.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(destination), "metrics": metrics}, ensure_ascii=False))


if __name__ == "__main__":
    main()
