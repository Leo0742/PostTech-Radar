from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import OneHotEncoder

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

from app.core.config import DATABASE_PATH  # noqa: E402
from app.ml.v3.candidates import CandidateSpec, build_estimator, safe_feature_row  # noqa: E402
from app.ml.v4.evaluation import REGISTRATION_FIELDS, classification_metrics  # noqa: E402
from app.services.data_service import load_rows  # noqa: E402

META_FIELDS = tuple(field for field in REGISTRATION_FIELDS if field != "description")


def _category_predictions(train: list[dict[str, Any]], validation: list[dict[str, Any]], seed: int) -> tuple[list[str], list[str]]:
    groups = [str(row.get("normalized_description") or row["request_id"]) for row in train]
    labels = [str(row["category"]) for row in train]
    splitter = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=seed)
    train_oof = [""] * len(train)
    indices = np.arange(len(train))
    for inner_train, inner_validation in splitter.split(indices, labels, groups):
        model = build_estimator(CandidateSpec("routing/category-inner/v4", "category", "structured_lr", "combined"))
        model.fit([safe_feature_row(train[index]) for index in inner_train], [labels[index] for index in inner_train])
        values = model.predict([safe_feature_row(train[index]) for index in inner_validation])
        for index, value in zip(inner_validation, values, strict=True):
            train_oof[int(index)] = str(value)
    if not all(train_oof):
        raise RuntimeError("missing inner OOF category predictions")
    final = build_estimator(CandidateSpec("routing/category-outer/v4", "category", "structured_lr", "combined"))
    final.fit([safe_feature_row(row) for row in train], labels)
    validation_predictions = [str(value) for value in final.predict([safe_feature_row(row) for row in validation])]
    return train_oof, validation_predictions


def _routing_features(
    train: list[dict[str, Any]],
    validation: list[dict[str, Any]],
    train_categories: list[str] | None,
    validation_categories: list[str] | None,
) -> tuple[sparse.csr_matrix, sparse.csr_matrix]:
    word = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=24000, sublinear_tf=True)
    char = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, max_features=32000)
    train_text = [str(row.get("description") or "") for row in train]
    validation_text = [str(row.get("description") or "") for row in validation]
    train_parts = [word.fit_transform(train_text), char.fit_transform(train_text)]
    validation_parts = [word.transform(validation_text), char.transform(validation_text)]
    metadata_train = [[str(row.get(field) or "") for field in META_FIELDS] for row in train]
    metadata_validation = [[str(row.get(field) or "") for field in META_FIELDS] for row in validation]
    if train_categories is not None and validation_categories is not None:
        metadata_train = [values + [category] for values, category in zip(metadata_train, train_categories, strict=True)]
        metadata_validation = [
            values + [category] for values, category in zip(metadata_validation, validation_categories, strict=True)
        ]
    encoder = OneHotEncoder(handle_unknown="ignore", min_frequency=2)
    train_parts.append(encoder.fit_transform(metadata_train))
    validation_parts.append(encoder.transform(metadata_validation))
    return sparse.hstack(train_parts, format="csr"), sparse.hstack(validation_parts, format="csr")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    pointer = json.loads((PROJECT_ROOT / "artifacts/gpu_research_v4/protocol.json").read_text())
    protocol = json.loads((PROJECT_ROOT / pointer["protocol_path"]).read_text())
    rows = load_rows(DATABASE_PATH)
    development = set(protocol["development_request_ids"])
    by_id = {str(row["request_id"]): row for row in rows if str(row["request_id"]) in development}
    predictions: dict[str, list[dict[str, str]]] = {"baseline": [], "oof_category": []}
    audit = []
    for fold in protocol["folds"]:
        if int(fold["repeat"]) != 0:
            continue
        train = [by_id[item] for item in fold["train_request_ids"] if item in by_id]
        validation = [by_id[item] for item in fold["validation_request_ids"] if item in by_id]
        train_category, validation_category = _category_predictions(train, validation, int(fold["seed"]))
        for name, categories in (("baseline", None), ("oof_category", (train_category, validation_category))):
            train_matrix, validation_matrix = _routing_features(
                train,
                validation,
                None if categories is None else categories[0],
                None if categories is None else categories[1],
            )
            model = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=int(fold["seed"]))
            model.fit(train_matrix, [str(row["final_line"]) for row in train])
            values = model.predict(validation_matrix)
            predictions[name].extend(
                {
                    "request_id": str(row["request_id"]),
                    "truth": str(row["final_line"]),
                    "prediction": str(value),
                }
                for row, value in zip(validation, values, strict=True)
            )
        audit.append({"fold": fold["fold"], "train": len(train), "validation": len(validation), "inner_oof": len(train_category)})
    labels = sorted({str(row["final_line"]) for row in by_id.values()})
    metrics = {
        name: classification_metrics(
            [row["truth"] for row in values], [row["prediction"] for row in values], labels=labels
        )
        for name, values in predictions.items()
    }
    payload = {
        "candidate_id": "routing/structured-plus-oof-category/v4",
        "status": "MEASURED_STRICT_NESTED_OOF",
        "dataset_sha256": protocol["dataset_sha256"],
        "split_sha256": protocol["split_sha256"],
        "ground_truth_category_used_as_feature": False,
        "line4_reliable": False,
        "fold_audit": audit,
        "metrics": metrics,
        "predictions": predictions,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "metrics": metrics}, ensure_ascii=False))


if __name__ == "__main__":
    main()
