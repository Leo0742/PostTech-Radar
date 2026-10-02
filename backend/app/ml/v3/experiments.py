from __future__ import annotations

import importlib.metadata
import time
from dataclasses import asdict
from statistics import mean, pstdev
from typing import Any

import numpy as np
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score

from app.ml.v3.candidates import CandidateSpec, build_estimator, fit_group_calibrated_svc, safe_feature_row
from app.ml.v3.protocol import EvaluationManifest


def evaluate_embedding_candidate(
    *,
    candidate_id: str,
    model_id: str,
    revision: str,
    embeddings: np.ndarray,
    rows: list[dict[str, Any]],
    manifest: EvaluationManifest,
    target: str = "category",
    with_metadata: bool = False,
) -> dict[str, Any]:
    if len(embeddings) != len(rows):
        raise ValueError("embeddings and rows must have identical length")
    embedding_by_id = {str(row["request_id"]): embeddings[index] for index, row in enumerate(rows)}
    row_by_id = {str(row["request_id"]): row for row in rows}
    fold_metrics: list[dict[str, Any]] = []
    predictions: list[dict[str, Any]] = []
    fit_seconds = 0.0
    predict_seconds = 0.0
    for fold in manifest.folds:
        train_rows = [row_by_id[request_id] for request_id in fold.train_request_ids]
        validation_rows = [row_by_id[request_id] for request_id in fold.validation_request_ids]
        x_train_dense = np.vstack([embedding_by_id[request_id] for request_id in fold.train_request_ids])
        x_validation_dense = np.vstack([embedding_by_id[request_id] for request_id in fold.validation_request_ids])
        if with_metadata:
            vectorizer = DictVectorizer(sparse=True, sort=True)
            train_metadata = [
                {key: value for key, value in safe_feature_row(row).items() if key != "description"}
                for row in train_rows
            ]
            validation_metadata = [
                {key: value for key, value in safe_feature_row(row).items() if key != "description"}
                for row in validation_rows
            ]
            train_encoded = vectorizer.fit_transform(train_metadata)
            validation_encoded = vectorizer.transform(validation_metadata)
            x_train = hstack([csr_matrix(x_train_dense), train_encoded], format="csr")
            x_validation = hstack([csr_matrix(x_validation_dense), validation_encoded], format="csr")
        else:
            x_train = x_train_dense
            x_validation = x_validation_dense
        train_labels = [str(row[target]) for row in train_rows]
        validation_labels = [str(row[target]) for row in validation_rows]
        estimator = LogisticRegression(
            max_iter=2_000,
            C=3.0,
            class_weight="balanced",
            random_state=fold.seed + fold.fold,
        )
        started = time.perf_counter()
        estimator.fit(x_train, train_labels)
        fit_seconds += time.perf_counter() - started
        started = time.perf_counter()
        predicted = estimator.predict(x_validation)
        probabilities = estimator.predict_proba(x_validation)
        predict_seconds += time.perf_counter() - started
        fold_metrics.append(
            {
                "repeat": fold.repeat,
                "fold": fold.fold,
                "size": len(validation_labels),
                "accuracy": round(float(accuracy_score(validation_labels, predicted)), 6),
                "macro_f1": round(float(f1_score(validation_labels, predicted, average="macro", zero_division=0)), 6),
                "weighted_f1": round(float(f1_score(validation_labels, predicted, average="weighted", zero_division=0)), 6),
            }
        )
        for index, request_id in enumerate(fold.validation_request_ids):
            predictions.append(
                {
                    "repeat": fold.repeat,
                    "fold": fold.fold,
                    "request_id": request_id,
                    "truth": validation_labels[index],
                    "prediction": str(predicted[index]),
                    "probabilities": {
                        str(label): round(float(probabilities[index, position]), 8)
                        for position, label in enumerate(estimator.classes_)
                    },
                }
            )
    all_truth = [item["truth"] for item in predictions]
    all_predictions = [item["prediction"] for item in predictions]
    report = classification_report(all_truth, all_predictions, output_dict=True, zero_division=0)
    per_class = {
        label: {
            "precision": round(float(values["precision"]), 6),
            "recall": round(float(values["recall"]), 6),
            "f1": round(float(values["f1-score"]), 6),
            "support": int(values["support"]),
        }
        for label, values in report.items()
        if label not in {"accuracy", "macro avg", "weighted avg"}
    }
    macro_scores = [item["macro_f1"] for item in fold_metrics]
    accuracy_scores = [item["accuracy"] for item in fold_metrics]
    return {
        "candidate_id": candidate_id,
        "status": "measured",
        "tier": "A",
        "task": target,
        "family": "frozen_sentence_embedding_lr",
        "feature_mode": "embedding+metadata" if with_metadata else "embedding",
        "model_id": model_id,
        "revision": revision,
        "dataset_sha256": manifest.dataset_sha256,
        "split_sha256": manifest.split_sha256,
        "development_only": True,
        "sealed_holdout_accessed": False,
        "metrics": {
            "folds": len(fold_metrics),
            "cv_macro_f1_mean": round(mean(macro_scores), 6),
            "cv_macro_f1_std": round(pstdev(macro_scores), 6),
            "cv_accuracy_mean": round(mean(accuracy_scores), 6),
            "cv_accuracy_std": round(pstdev(accuracy_scores), 6),
            "oof_accuracy": round(float(accuracy_score(all_truth, all_predictions)), 6),
            "oof_macro_f1": round(float(f1_score(all_truth, all_predictions, average="macro", zero_division=0)), 6),
            "oof_weighted_f1": round(float(f1_score(all_truth, all_predictions, average="weighted", zero_division=0)), 6),
            "worst_class_f1": round(min((value["f1"] for value in per_class.values()), default=0.0), 6),
            "fit_seconds": round(fit_seconds, 3),
            "prediction_ms_per_ticket": round(1_000 * predict_seconds / max(1, len(predictions)), 4),
            "per_class": per_class,
            "fold_metrics": fold_metrics,
        },
        "predictions": predictions,
    }


def evaluate_candidate(
    spec: CandidateSpec,
    rows: list[dict[str, Any]],
    manifest: EvaluationManifest,
    *,
    target: str = "category",
) -> dict[str, Any]:
    by_id = {str(row["request_id"]): row for row in rows}
    fold_metrics: list[dict[str, Any]] = []
    predictions: list[dict[str, Any]] = []
    fit_seconds = 0.0
    predict_seconds = 0.0
    calibration_metadata: list[dict[str, Any]] = []
    for fold in manifest.folds:
        train_rows = [by_id[request_id] for request_id in fold.train_request_ids]
        validation_rows = [by_id[request_id] for request_id in fold.validation_request_ids]
        train_values = [safe_feature_row(row) for row in train_rows]
        validation_values = [safe_feature_row(row) for row in validation_rows]
        train_labels = [str(row[target]) for row in train_rows]
        validation_labels = [str(row[target]) for row in validation_rows]
        started = time.perf_counter()
        if spec.family == "calibrated_linear_svc":
            estimator, calibration = fit_group_calibrated_svc(
                spec,
                train_values,
                train_labels,
                [str(row.get("normalized_description") or row["request_id"]) for row in train_rows],
                seed=fold.seed + fold.fold,
            )
            calibration_metadata.append({"repeat": fold.repeat, "fold": fold.fold, **calibration})
        else:
            estimator = build_estimator(spec)
            estimator.fit(train_values, train_labels)
        fit_seconds += time.perf_counter() - started
        started = time.perf_counter()
        predicted = estimator.predict(validation_values)
        probabilities = estimator.predict_proba(validation_values) if hasattr(estimator, "predict_proba") else None
        predict_seconds += time.perf_counter() - started
        macro = float(f1_score(validation_labels, predicted, average="macro", zero_division=0))
        weighted = float(f1_score(validation_labels, predicted, average="weighted", zero_division=0))
        fold_metrics.append(
            {
                "repeat": fold.repeat,
                "fold": fold.fold,
                "size": len(validation_labels),
                "accuracy": round(float(accuracy_score(validation_labels, predicted)), 6),
                "macro_f1": round(macro, 6),
                "weighted_f1": round(weighted, 6),
            }
        )
        classes = [str(value) for value in estimator.classes_] if probabilities is not None else []
        for index, request_id in enumerate(fold.validation_request_ids):
            item = {
                "repeat": fold.repeat,
                "fold": fold.fold,
                "request_id": request_id,
                "truth": validation_labels[index],
                "prediction": str(predicted[index]),
            }
            if probabilities is not None:
                item["probabilities"] = {label: round(float(probabilities[index, position]), 8) for position, label in enumerate(classes)}
            predictions.append(item)

    macro_scores = [item["macro_f1"] for item in fold_metrics]
    accuracy_scores = [item["accuracy"] for item in fold_metrics]
    all_truth = [item["truth"] for item in predictions]
    all_predictions = [item["prediction"] for item in predictions]
    report = classification_report(all_truth, all_predictions, output_dict=True, zero_division=0)
    per_class = {
        label: {
            "precision": round(float(values["precision"]), 6),
            "recall": round(float(values["recall"]), 6),
            "f1": round(float(values["f1-score"]), 6),
            "support": int(values["support"]),
        }
        for label, values in report.items()
        if label not in {"accuracy", "macro avg", "weighted avg"}
    }
    metrics = {
        "folds": len(fold_metrics),
        "cv_macro_f1_mean": round(mean(macro_scores), 6),
        "cv_macro_f1_std": round(pstdev(macro_scores), 6),
        "cv_accuracy_mean": round(mean(accuracy_scores), 6),
        "cv_accuracy_std": round(pstdev(accuracy_scores), 6),
        "oof_accuracy": round(float(accuracy_score(all_truth, all_predictions)), 6),
        "oof_macro_f1": round(float(f1_score(all_truth, all_predictions, average="macro", zero_division=0)), 6),
        "oof_weighted_f1": round(float(f1_score(all_truth, all_predictions, average="weighted", zero_division=0)), 6),
        "worst_class_f1": round(min((values["f1"] for values in per_class.values()), default=0.0), 6),
        "fit_seconds": round(fit_seconds, 3),
        "prediction_ms_per_ticket": round(1_000 * predict_seconds / max(1, len(predictions)), 4),
        "per_class": per_class,
        "fold_metrics": fold_metrics,
    }
    return {
        "candidate_id": spec.candidate_id,
        "status": "measured",
        "tier": spec.tier,
        "task": spec.task,
        "family": spec.family,
        "feature_mode": spec.feature_mode,
        "parameters": dict(spec.parameters),
        "dataset_sha256": manifest.dataset_sha256,
        "split_sha256": manifest.split_sha256,
        "development_only": True,
        "sealed_holdout_accessed": False,
        "package_versions": {
            name: importlib.metadata.version(name)
            for name in ("scikit-learn", "numpy")
        },
        "metrics": metrics,
        "calibration": calibration_metadata,
        "predictions": predictions,
        "spec": asdict(spec),
    }
