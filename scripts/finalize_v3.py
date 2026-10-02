from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

import joblib
import numpy as np
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    log_loss,
    roc_auc_score,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

from app.core.config import DATABASE_PATH  # noqa: E402
from app.ml.v3.candidates import CandidateSpec, build_estimator, safe_feature_row  # noqa: E402
from app.ml.v3.conformal import APSConformalClassifier  # noqa: E402
from app.ml.v3.dataset import challenge_dataset_config, training_corpus_summary  # noqa: E402
from app.ml.v3.metrics import paired_bootstrap_accuracy, risk_coverage_curve  # noqa: E402
from app.ml.v3.oos import probability_oos_scores, threshold_for_oos_recall  # noqa: E402
from app.ml.v3.protocol import manifest_from_dict  # noqa: E402
from app.ml.v3.selective import coverage_at_accuracy_targets, learn_class_thresholds  # noqa: E402
from app.services.data_service import load_rows  # noqa: E402


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _matrix(predictions: list[dict], labels: list[str]):
    keyed = {(item["repeat"], item["fold"], item["request_id"]): item for item in predictions}
    keys = sorted(keyed)
    truth = [keyed[key]["truth"] for key in keys]
    matrix = np.asarray([[keyed[key]["probabilities"].get(label, 0.0) for label in labels] for key in keys])
    return keys, truth, matrix


def _fit_embedding_model(train_rows: list[dict], embeddings: np.ndarray, labels: list[str]):
    vectorizer = DictVectorizer(sparse=True, sort=True)
    metadata = [
        {key: value for key, value in safe_feature_row(row).items() if key != "description"}
        for row in train_rows
    ]
    x = hstack([csr_matrix(embeddings), vectorizer.fit_transform(metadata)], format="csr")
    model = LogisticRegression(max_iter=2_000, C=3.0, class_weight="balanced", random_state=20260916)
    model.fit(x, [str(row["category"]) for row in train_rows])
    if set(model.classes_) != set(labels):
        raise ValueError("training partition does not contain all labels")
    return model, vectorizer


def _predict_embedding(model, vectorizer, rows: list[dict], embeddings: np.ndarray, labels: list[str]) -> np.ndarray:
    metadata = [
        {key: value for key, value in safe_feature_row(row).items() if key != "description"}
        for row in rows
    ]
    x = hstack([csr_matrix(embeddings), vectorizer.transform(metadata)], format="csr")
    raw = model.predict_proba(x)
    positions = {str(label): index for index, label in enumerate(model.classes_)}
    return np.asarray([[row[positions[label]] for label in labels] for row in raw])


def _predict_structured(model, rows: list[dict], labels: list[str]) -> np.ndarray:
    raw = model.predict_proba([safe_feature_row(row) for row in rows])
    positions = {str(label): index for index, label in enumerate(model.classes_)}
    return np.asarray([[row[positions[label]] for label in labels] for row in raw])


def _classification_metrics(truth: list[str], probabilities: np.ndarray, labels: list[str]) -> dict:
    predicted = np.asarray(labels)[probabilities.argmax(axis=1)]
    report = classification_report(truth, predicted, labels=labels, output_dict=True, zero_division=0)
    one_hot = np.zeros_like(probabilities)
    label_to_index = {label: index for index, label in enumerate(labels)}
    for row, label in enumerate(truth):
        one_hot[row, label_to_index[label]] = 1.0
    confidence = probabilities.max(axis=1)
    correct = predicted == np.asarray(truth)
    bins = np.linspace(0, 1, 11)
    ece = 0.0
    for lower, upper in zip(bins[:-1], bins[1:], strict=True):
        mask = (confidence >= lower) & (confidence < upper if upper < 1 else confidence <= upper)
        if mask.any():
            ece += float(mask.mean()) * abs(float(correct[mask].mean()) - float(confidence[mask].mean()))
    return {
        "accuracy": round(float(accuracy_score(truth, predicted)), 6),
        "macro_f1": round(float(f1_score(truth, predicted, average="macro", zero_division=0)), 6),
        "weighted_f1": round(float(f1_score(truth, predicted, average="weighted", zero_division=0)), 6),
        "worst_class_f1": round(min(float(report[label]["f1-score"]) for label in labels), 6),
        "ece_10_bin": round(ece, 6),
        "multiclass_brier": round(float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1))), 6),
        "nll": round(float(log_loss(truth, probabilities, labels=labels)), 6),
        "per_class": {
            label: {
                "precision": round(float(report[label]["precision"]), 6),
                "recall": round(float(report[label]["recall"]), 6),
                "f1": round(float(report[label]["f1-score"]), 6),
                "support": int(report[label]["support"]),
            }
            for label in labels
        },
    }


def _evaluate_thresholds(truth: list[str], probabilities: np.ndarray, labels: list[str], thresholds: dict[str, float]) -> dict:
    predictions = np.asarray(labels)[probabilities.argmax(axis=1)]
    confidence = probabilities.max(axis=1)
    accepted = np.asarray([confidence[index] >= thresholds[prediction] for index, prediction in enumerate(predictions)])
    return {
        "coverage": round(float(accepted.mean()), 6),
        "accepted": int(accepted.sum()),
        "accepted_accuracy": round(float((predictions[accepted] == np.asarray(truth)[accepted]).mean()), 6) if accepted.any() else 0.0,
        "errors": int((predictions[accepted] != np.asarray(truth)[accepted]).sum()),
    }


def _rule_experiment(keys, truth, base_predictions, rows_by_id):
    rules = [
        (r"(?:ф\s*\.?\s*103|zip)", "Импорт списков из архива ф.103 (zip-архив)"),
        (r"лк\s+юл", "Импорт списков из ЛК ЮЛ"),
        (r"(?:pre\s*post|пре\s*пост)", "Проблемы в работе Препост/PrePost"),
        (r"\bqr\b", "Проблема с QR-код(подключение/отключение)"),
        (r"(?:\bpush\b|\bsms\b|e-?mail)", "Проблема с push/sms/email"),
    ]
    changed = np.asarray(base_predictions).copy()
    fired = []
    for index, key in enumerate(keys):
        text = str(rows_by_id[key[2]]["description"]).lower()
        for pattern, label in rules:
            if re.search(pattern, text, re.IGNORECASE):
                changed[index] = label
                fired.append((index, label))
                break
    precision = sum(truth[index] == label for index, label in fired) / len(fired) if fired else 0.0
    return {
        "strategy": "high_precision_domain_rule_override_candidate",
        "fired": len(fired),
        "rule_precision": round(precision, 6),
        "base_macro_f1": round(float(f1_score(truth, base_predictions, average="macro", zero_division=0)), 6),
        "candidate_macro_f1": round(float(f1_score(truth, changed, average="macro", zero_division=0)), 6),
        "decision": "REJECTED_NO_GENERALIZATION_GAIN",
    }


def main() -> None:
    final_path = PROJECT_ROOT / "artifacts/evaluation/v3/final_holdout.json"
    if final_path.exists():
        print(json.dumps({"status": "REFUSED_ALREADY_EVALUATED", "artifact": str(final_path)}, ensure_ascii=False))
        return
    rows = [row for row in load_rows(DATABASE_PATH) if str(row.get("description") or "").strip()]
    top = {item["name"] for item in training_corpus_summary(DATABASE_PATH, challenge_dataset_config())["top_k"]}
    category_rows = [row for row in rows if row["category"] in top]
    oos_rows = [row for row in rows if row["category"] not in top]
    rows_by_id = {str(row["request_id"]): row for row in category_rows}
    manifest = manifest_from_dict(json.loads((PROJECT_ROOT / "artifacts/evaluation/v3/protocol.json").read_text(encoding="utf-8")))
    labels = list(manifest.labels)

    structured_result = json.loads((PROJECT_ROOT / "artifacts/research/v3/candidates/category-structured_lr.json").read_text(encoding="utf-8"))
    gte_result = json.loads((PROJECT_ROOT / "artifacts/research/v3/embeddings/gte-multilingual-base.json").read_text(encoding="utf-8"))
    gte_category = gte_result["category"][1]
    structured_keys, truth, structured_oof = _matrix(structured_result["predictions"], labels)
    gte_keys, gte_truth, gte_oof = _matrix(gte_category["predictions"], labels)
    if structured_keys != gte_keys or truth != gte_truth:
        raise ValueError("OOF predictions are not paired")
    tuning = []
    for structured_weight in np.linspace(0, 1, 11):
        probabilities = structured_weight * structured_oof + (1 - structured_weight) * gte_oof
        predicted = np.asarray(labels)[probabilities.argmax(axis=1)]
        tuning.append(
            {
                "structured_weight": round(float(structured_weight), 1),
                "gte_weight": round(float(1 - structured_weight), 1),
                "macro_f1": round(float(f1_score(truth, predicted, average="macro", zero_division=0)), 6),
                "accuracy": round(float(accuracy_score(truth, predicted)), 6),
            }
        )
    selected = max(tuning, key=lambda item: (item["macro_f1"], item["accuracy"], -item["structured_weight"]))
    structured_weight = selected["structured_weight"]
    gte_weight = selected["gte_weight"]
    ensemble_oof = structured_weight * structured_oof + gte_weight * gte_oof
    ensemble_predictions = np.asarray(labels)[ensemble_oof.argmax(axis=1)]
    rules = _rule_experiment(structured_keys, truth, ensemble_predictions, rows_by_id)
    development = {
        "weight_tuning": tuning,
        "selected_weights": selected,
        "metrics": _classification_metrics(truth, ensemble_oof, labels),
        "paired_bootstrap_vs_gte": paired_bootstrap_accuracy(
            truth,
            ensemble_predictions,
            np.asarray(labels)[gte_oof.argmax(axis=1)],
        ),
        "risk_coverage": risk_coverage_curve(truth, ensemble_oof, labels),
        "coverage_at_accuracy_targets": coverage_at_accuracy_targets(truth, ensemble_oof, labels),
        "class_thresholds_95": learn_class_thresholds(truth, ensemble_oof, labels, target_accuracy=0.95),
        "confusion_pair_strategy": rules,
    }

    all_position = {str(row["request_id"]): index for index, row in enumerate(rows)}
    embedding_root = PROJECT_ROOT / "artifacts/research/v3/embedding-cache"
    classification_key = gte_result["encoding"]["classification"]["cache_key"]
    all_embeddings = np.load(embedding_root / f"{classification_key}.npy", allow_pickle=False)
    development_rows = [rows_by_id[value] for value in manifest.development_request_ids]
    calibration_rows = [rows_by_id[value] for value in manifest.calibration_request_ids]
    holdout_rows = [rows_by_id[value] for value in manifest.holdout_request_ids]
    def embeddings_for(selected_rows):
        return all_embeddings[[all_position[str(row["request_id"])] for row in selected_rows]]

    embedding_model, metadata_vectorizer = _fit_embedding_model(development_rows, embeddings_for(development_rows), labels)
    structured_model = build_estimator(CandidateSpec("category/structured_lr/final", "category", "structured_lr", "combined"))
    structured_model.fit([safe_feature_row(row) for row in development_rows], [row["category"] for row in development_rows])
    def ensemble_for(selected_rows):
        return (
            structured_weight * _predict_structured(structured_model, selected_rows, labels)
            + gte_weight * _predict_embedding(embedding_model, metadata_vectorizer, selected_rows, embeddings_for(selected_rows), labels)
        )
    calibration_probabilities = ensemble_for(calibration_rows)
    holdout_probabilities = ensemble_for(holdout_rows)
    calibration_truth = [str(row["category"]) for row in calibration_rows]
    holdout_truth = [str(row["category"]) for row in holdout_rows]
    thresholds = learn_class_thresholds(calibration_truth, calibration_probabilities, labels, target_accuracy=0.95, minimum_predictions=3)
    conformal = []
    for alpha in (0.1, 0.05, 0.025, 0.01):
        estimator = APSConformalClassifier(alpha=alpha).fit(calibration_truth, calibration_probabilities, labels)
        conformal.append(estimator.summary(holdout_truth, holdout_probabilities))

    oos_grouped = sorted(
        oos_rows,
        key=lambda row: hashlib.sha256(str(row.get("normalized_description") or row["request_id"]).encode()).hexdigest(),
    )
    oos_calibration = oos_grouped[::2]
    oos_test = oos_grouped[1::2]
    oos_calibration_probabilities = ensemble_for(oos_calibration)
    oos_test_probabilities = ensemble_for(oos_test)
    combined_calibration = np.vstack([calibration_probabilities, oos_calibration_probabilities])
    combined_test = np.vstack([holdout_probabilities, oos_test_probabilities])
    calibration_is_oos = [False] * len(calibration_rows) + [True] * len(oos_calibration)
    test_is_oos = np.asarray([False] * len(holdout_rows) + [True] * len(oos_test))
    calibration_scores = probability_oos_scores(combined_calibration)
    test_scores = probability_oos_scores(combined_test)
    methods = {
        "one_minus_max_probability": [1 - item["maximum_probability"] for item in calibration_scores],
        "one_minus_margin": [1 - item["margin"] for item in calibration_scores],
        "normalized_entropy": [item["normalized_entropy"] for item in calibration_scores],
    }
    test_methods = {
        "one_minus_max_probability": np.asarray([1 - item["maximum_probability"] for item in test_scores]),
        "one_minus_margin": np.asarray([1 - item["margin"] for item in test_scores]),
        "normalized_entropy": np.asarray([item["normalized_entropy"] for item in test_scores]),
    }
    oos_results = {}
    for name, scores in methods.items():
        threshold = threshold_for_oos_recall(scores, calibration_is_oos, target_recall=0.9)
        predicted_oos = test_methods[name] >= threshold
        oos_results[name] = {
            "threshold": threshold,
            "auroc": round(float(roc_auc_score(test_is_oos, test_methods[name])), 6),
            "oos_rejection_recall": round(float(predicted_oos[test_is_oos].mean()), 6),
            "oos_false_acceptance": round(float((~predicted_oos[test_is_oos]).mean()), 6),
            "in_scope_false_rejection": round(float(predicted_oos[~test_is_oos].mean()), 6),
            "oos_test_count": int(test_is_oos.sum()),
            "in_scope_test_count": int((~test_is_oos).sum()),
        }

    final = {
        "sealed_holdout_opened_once": True,
        "frozen_configuration": {
            "structured_weight": structured_weight,
            "gte_weight": gte_weight,
            "gte_model_id": gte_result["model_id"],
            "gte_revision": gte_result["revision"],
            "class_thresholds": thresholds,
        },
        "holdout": {
            "size": len(holdout_rows),
            "metrics": _classification_metrics(holdout_truth, holdout_probabilities, labels),
            "risk_coverage": risk_coverage_curve(holdout_truth, holdout_probabilities, labels),
            "coverage_at_accuracy_targets": coverage_at_accuracy_targets(holdout_truth, holdout_probabilities, labels),
            "class_specific_thresholds_95": _evaluate_thresholds(holdout_truth, holdout_probabilities, labels, thresholds),
            "conformal": conformal,
        },
        "oos": oos_results,
    }
    _write(PROJECT_ROOT / "artifacts/evaluation/v3/development_selection.json", development)
    _write(final_path, final)

    deploy_rows = category_rows
    deploy_embeddings = embeddings_for(deploy_rows)
    deploy_embedding_model, deploy_vectorizer = _fit_embedding_model(deploy_rows, deploy_embeddings, labels)
    deploy_structured = build_estimator(CandidateSpec("category/structured_lr/deployment", "category", "structured_lr", "combined"))
    deploy_structured.fit([safe_feature_row(row) for row in deploy_rows], [row["category"] for row in deploy_rows])
    model_dir = PROJECT_ROOT / "artifacts/models/v3/category-ensemble"
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(deploy_structured, model_dir / "structured.joblib")
    joblib.dump(deploy_embedding_model, model_dir / "embedding-head.joblib")
    joblib.dump(deploy_vectorizer, model_dir / "metadata-vectorizer.joblib")
    _write(
        model_dir / "metadata.json",
        {
            "status": "deployment_model_trained_on_all_eligible_labeled_history",
            "training_count": len(deploy_rows),
            "labels": labels,
            "dataset_sha256": manifest.dataset_sha256,
            "weights": {"structured": structured_weight, "gte": gte_weight},
            "encoder": {"model_id": gte_result["model_id"], "revision": gte_result["revision"], "not_bundled": True},
            "class_thresholds_from_calibration": thresholds,
        },
    )
    print(json.dumps({"development": development["selected_weights"], "final": final["holdout"]["metrics"], "oos": oos_results}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
