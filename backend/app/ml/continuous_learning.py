from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import FeatureUnion, Pipeline

from app.ml.training import registration_text
from app.services.data_service import ensure_database_schema, normalize_description

FEATURE_FIELDS = (
    "description",
    "registration_date",
    "user",
    "service",
    "component",
    "request_type",
    "criticality",
    "urgency",
    "priority",
    "service_class",
    "timezone",
)


def _features(values: dict[str, Any]) -> dict[str, str]:
    result: dict[str, str] = {}
    for field in FEATURE_FIELDS:
        source = "user_name" if field == "user" else field
        value = values.get(source, values.get(field, ""))
        result[field] = str(value or "").strip()
    return result


def _group_key(features: dict[str, str], request_id: str) -> str:
    normalized = normalize_description(features.get("description"))
    basis = normalized or f"id:{request_id}"
    return hashlib.sha256(basis.encode("utf-8")).hexdigest()


def build_retraining_dataset(database: Path) -> dict[str, Any]:
    """Build a leakage-safe manual retraining dataset.

    Historical imported rows and operator-confirmed production feedback stay tagged by
    source. Only human-confirmed labels are accepted from feedback; model predictions are
    never copied into the feature contract.
    """
    ensure_database_schema(database)
    connection = sqlite3.connect(database)
    connection.row_factory = sqlite3.Row

    rows_by_id: dict[str, dict[str, Any]] = {}
    historical_count = 0
    for row in connection.execute(
        "SELECT * FROM tickets WHERE source_hash <> 'operator-review' ORDER BY request_id"
    ).fetchall():
        item = dict(row)
        request_id = str(item.get("request_id") or "").strip()
        category = str(item.get("category") or "").strip()
        final_line = str(item.get("final_line") or "").strip()
        if not request_id or not category or not final_line:
            continue
        features = _features(item)
        rows_by_id[request_id] = {
            "request_id": request_id,
            "features": features,
            "category": category,
            "final_line": final_line,
            "group_key": _group_key(features, request_id),
            "source": "historical",
        }
        historical_count += 1

    feedback_count = 0
    for row in connection.execute(
        "SELECT * FROM production_feedback ORDER BY confirmation_timestamp, id"
    ).fetchall():
        item = dict(row)
        request_id = str(item.get("request_id") or "").strip()
        category = str(item.get("operator_final_category") or "").strip()
        final_line = str(item.get("operator_final_route") or "").strip()
        if not request_id or not category or not final_line or not item.get("confirmation_timestamp"):
            continue
        registration = json.loads(item.get("registration_json") or "{}")
        features = _features(registration)
        rows_by_id[request_id] = {
            "request_id": request_id,
            "features": features,
            "category": category,
            "final_line": final_line,
            "group_key": _group_key(features, request_id),
            "source": "feedback",
            "confirmation_timestamp": item["confirmation_timestamp"],
        }
        feedback_count += 1
    connection.close()

    rows = list(rows_by_id.values())
    rows.sort(key=lambda item: (item["source"] != "historical", item["request_id"]))
    return {
        "schema_version": "6.0",
        "feature_contract": list(FEATURE_FIELDS),
        "historical_count": historical_count,
        "feedback_count": feedback_count,
        "rows_total": len(rows),
        "rows": rows,
    }

def _pipeline() -> Pipeline:
    return Pipeline(
        [
            (
                "features",
                FeatureUnion(
                    [
                        ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_features=30_000, sublinear_tf=True)),
                        (
                            "char",
                            TfidfVectorizer(
                                analyzer="char_wb", ngram_range=(3, 5), min_df=1,
                                max_features=50_000, sublinear_tf=True,
                            ),
                        ),
                    ]
                ),
            ),
            (
                "classifier",
                LogisticRegression(max_iter=3000, class_weight="balanced", random_state=20260918),
            ),
        ]
    )


def _split_indices(rows: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    indices = np.arange(len(rows))
    groups = np.asarray([row["group_key"] for row in rows])
    labels = np.asarray([row["category"] for row in rows])
    routes = np.asarray([row["final_line"] for row in rows])
    for seed in range(20260918, 20260938):
        train, validation = next(
            GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=seed).split(indices, labels, groups)
        )
        if len(set(labels[train])) >= 2 and len(set(routes[train])) >= 2:
            return train, validation
    raise ValueError("Unable to create a group-safe split with at least two category and routing classes")


def train_challenger(rows: list[dict[str, Any]], output: Path) -> dict[str, Any]:
    if len(rows) < 8:
        raise ValueError("At least 8 confirmed rows are required to train a challenger")
    train_idx, validation_idx = _split_indices(rows)
    texts = [registration_text(row["features"]) for row in rows]
    categories = np.asarray([row["category"] for row in rows])
    routes = np.asarray([row["final_line"] for row in rows])

    category_model = _pipeline()
    category_model.fit([texts[index] for index in train_idx], categories[train_idx])
    category_probabilities = category_model.predict_proba([texts[index] for index in validation_idx])
    category_predictions = category_model.classes_[np.argmax(category_probabilities, axis=1)]
    top_k = min(3, category_probabilities.shape[1])
    top_order = np.argsort(category_probabilities, axis=1)[:, ::-1][:, :top_k]
    top3_hits = []
    for row_index, target in enumerate(categories[validation_idx]):
        candidates = {str(category_model.classes_[index]) for index in top_order[row_index]}
        top3_hits.append(str(target) in candidates)

    allowed_routes = {"(1 линия)", "(2 линия)", "(3 линия)"}
    route_rows = [index for index in train_idx if routes[index] in allowed_routes]
    route_validation = [index for index in validation_idx if routes[index] in allowed_routes]
    if len(set(routes[route_rows])) < 2 or not route_validation:
        raise ValueError("Not enough routing labels for challenger evaluation")
    routing_model = _pipeline()
    routing_model.fit([texts[index] for index in route_rows], routes[route_rows])
    routing_predictions = routing_model.predict([texts[index] for index in route_validation])

    artifact = {
        "schema_version": "6.0",
        "category_pipeline": category_model,
        "routing_pipeline": routing_model,
        "feature_contract": list(FEATURE_FIELDS),
        "split_strategy": "GroupShuffleSplit by normalized-description group",
        "training_request_ids": [rows[index]["request_id"] for index in train_idx],
        "validation_request_ids": [rows[index]["request_id"] for index in validation_idx],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, output, compress=3)
    return {
        "split_strategy": artifact["split_strategy"],
        "category": {
            "training_size": int(len(train_idx)),
            "validation_size": int(len(validation_idx)),
            "top1": round(float(accuracy_score(categories[validation_idx], category_predictions)), 6),
            "top3": round(float(np.mean(top3_hits)), 6),
            "macro_f1": round(float(f1_score(categories[validation_idx], category_predictions, average="macro", zero_division=0)), 6),
        },
        "routing": {
            "training_size": len(route_rows),
            "validation_size": len(route_validation),
            "top1": round(float(accuracy_score(routes[route_validation], routing_predictions)), 6),
            "macro_f1": round(float(f1_score(routes[route_validation], routing_predictions, average="macro", zero_division=0)), 6),
        },
    }
