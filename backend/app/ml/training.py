from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import shutil
import time
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

from app.core.config import RANDOM_SEED
from app.services.data_service import load_rows, metadata_value


@dataclass(frozen=True)
class DatasetSplit:
    train: list[int]
    validation: list[int]
    test: list[int]


@dataclass(frozen=True)
class CanonicalSplit:
    development: list[int]
    test: list[int]
    cv_folds: list[tuple[list[int], list[int]]]


class ProbabilityModel:
    """Applies a temperature learned only from development OOF predictions."""

    def __init__(self, estimator: Any, temperature: float = 1.0):
        self.estimator = estimator
        self.temperature = temperature

    @property
    def classes_(self) -> np.ndarray:
        return self.estimator.named_steps["classifier"].classes_

    def predict_proba(self, values: Sequence[str]) -> np.ndarray:
        raw = np.clip(self.estimator.predict_proba(values), 1e-9, 1.0)
        adjusted = raw ** (1.0 / self.temperature)
        return adjusted / adjusted.sum(axis=1, keepdims=True)

    def predict(self, values: Sequence[str]) -> np.ndarray:
        probabilities = self.predict_proba(values)
        return self.classes_[probabilities.argmax(axis=1)]


def make_canonical_split(
    labels: list[str], groups: list[str], seed: int = RANDOM_SEED, outer_splits: int = 5, inner_splits: int = 4
) -> CanonicalSplit:
    indices = np.arange(len(labels))
    outer_splits = min(outer_splits, min(Counter(labels).values()))
    outer = StratifiedGroupKFold(n_splits=outer_splits, shuffle=True, random_state=seed)
    development, test = next(outer.split(indices, labels, groups))
    dev_labels = np.asarray(labels)[development]
    dev_groups = np.asarray(groups)[development]
    inner_splits = min(inner_splits, min(Counter(dev_labels.tolist()).values()))
    inner = StratifiedGroupKFold(n_splits=inner_splits, shuffle=True, random_state=seed + 1)
    folds = [(train.tolist(), validation.tolist()) for train, validation in inner.split(development, dev_labels, dev_groups)]
    return CanonicalSplit(development.tolist(), test.tolist(), folds)


def make_group_split(labels: list[str], groups: list[str], seed: int = RANDOM_SEED) -> DatasetSplit:
    canonical = make_canonical_split(labels, groups, seed)
    train_relative, validation_relative = canonical.cv_folds[0]
    return DatasetSplit(
        [canonical.development[index] for index in train_relative],
        [canonical.development[index] for index in validation_relative],
        canonical.test,
    )


def registration_text(row: dict[str, Any], mode: str = "combined") -> str:
    text_fields = (("описание", row.get("description")),)
    metadata_fields = (
        ("услуга", row.get("service")), ("компонент", row.get("component")),
        ("тип", row.get("request_type")), ("критичность", row.get("criticality")),
        ("срочность", row.get("urgency")), ("приоритет", row.get("priority")),
        ("класс", row.get("service_class")), ("часовой_пояс", row.get("timezone")),
    )
    fields = text_fields if mode == "text" else metadata_fields if mode == "metadata" else text_fields + metadata_fields
    return " ".join(f"{name}_{str(value).strip()}" for name, value in fields if value)


def _word_vectorizer(*, tuned: bool = False) -> TfidfVectorizer:
    return TfidfVectorizer(
        lowercase=True, ngram_range=(1, 3 if tuned else 2), min_df=2, max_df=0.995 if tuned else 0.98,
        max_features=20_000 if tuned else 14_000, sublinear_tf=True, strip_accents=None,
    )


def _char_vectorizer(*, tuned: bool = False) -> TfidfVectorizer:
    return TfidfVectorizer(
        analyzer="char_wb", ngram_range=(2, 6) if tuned else (3, 5), min_df=2,
        max_features=28_000 if tuned else 18_000, sublinear_tf=True,
    )


def _word_char_features(*, tuned: bool = False) -> FeatureUnion:
    return FeatureUnion([("word", _word_vectorizer(tuned=tuned)), ("char", _char_vectorizer(tuned=tuned))])


def _pipeline(kind: str) -> Pipeline:
    if kind == "word_lr":
        features: Any = _word_vectorizer()
        classifier: Any = LogisticRegression(max_iter=1000, class_weight="balanced", C=1.0, random_state=RANDOM_SEED)
    elif kind == "word_char_lr":
        features = _word_char_features()
        classifier = LogisticRegression(max_iter=1000, class_weight="balanced", C=1.0, random_state=RANDOM_SEED)
    elif kind == "tuned_word_char_lr":
        features = _word_char_features(tuned=True)
        classifier = LogisticRegression(max_iter=1200, class_weight="balanced", C=2.0, random_state=RANDOM_SEED)
    elif kind == "linear_svc":
        features = _word_char_features(tuned=True)
        classifier = LinearSVC(class_weight="balanced", C=0.8, random_state=RANDOM_SEED)
    else:
        raise ValueError(f"Неизвестная модель: {kind}")
    return Pipeline([("features", features), ("classifier", classifier)])


def _metrics(y_true: list[str], y_pred: list[str], labels: list[str]) -> dict[str, Any]:
    report = classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)
    per_class = {
        label: {"precision": round(report[label]["precision"], 4), "recall": round(report[label]["recall"], 4),
                "f1": round(report[label]["f1-score"], 4), "support": int(report[label]["support"])}
        for label in labels if label in report
    }
    return {
        "test_size": len(y_true), "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "macro_f1": round(f1_score(y_true, y_pred, average="macro", zero_division=0), 4),
        "weighted_f1": round(f1_score(y_true, y_pred, average="weighted", zero_division=0), 4),
        "worst_class_f1": round(min((item["f1"] for item in per_class.values()), default=0.0), 4),
        "per_class": per_class, "labels": labels,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }


def _save_confusion(metrics: dict[str, Any], path: Path, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    matrix = np.asarray(metrics["confusion_matrix"])
    labels = metrics["labels"]
    fig, axis = plt.subplots(figsize=(11, 9))
    image = axis.imshow(matrix, cmap="Blues")
    axis.set_title(title)
    axis.set_xlabel("Предсказано")
    axis.set_ylabel("Фактически")
    axis.set_xticks(range(len(labels)), [str(index + 1) for index in range(len(labels))])
    axis.set_yticks(range(len(labels)), [str(index + 1) for index in range(len(labels))])
    fig.colorbar(image, ax=axis, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def _class_probabilities(model: Pipeline, values: list[str], labels: list[str]) -> np.ndarray:
    probabilities = model.predict_proba(values)
    current = model.named_steps["classifier"].classes_.tolist()
    aligned = np.zeros((len(values), len(labels)))
    for source, label in enumerate(current):
        aligned[:, labels.index(label)] = probabilities[:, source]
    return aligned


def _temperature_scale(probabilities: np.ndarray, temperature: float) -> np.ndarray:
    adjusted = np.clip(probabilities, 1e-9, 1.0) ** (1.0 / temperature)
    return adjusted / adjusted.sum(axis=1, keepdims=True)


def _multiclass_brier(probabilities: np.ndarray, y_true: list[str], labels: list[str]) -> float:
    target = np.zeros_like(probabilities)
    for index, label in enumerate(y_true):
        target[index, labels.index(label)] = 1.0
    return float(np.mean(np.sum((probabilities - target) ** 2, axis=1)))


def _ece(probabilities: np.ndarray, y_true: list[str], labels: list[str], bins: int = 10) -> float:
    confidence = probabilities.max(axis=1)
    predicted = np.asarray(labels)[probabilities.argmax(axis=1)]
    correct = predicted == np.asarray(y_true)
    value = 0.0
    for lower in np.linspace(0, 1, bins, endpoint=False):
        upper = lower + 1 / bins
        mask = (confidence >= lower) & (confidence < upper if upper < 1 else confidence <= upper)
        if mask.any():
            value += float(mask.mean()) * abs(float(correct[mask].mean()) - float(confidence[mask].mean()))
    return value


def _fit_temperature(probabilities: np.ndarray, y_true: list[str], labels: list[str]) -> float:
    target = np.asarray([labels.index(label) for label in y_true])
    best_temperature, best_loss = 1.0, float("inf")
    for temperature in np.arange(0.6, 2.01, 0.1):
        scaled = _temperature_scale(probabilities, float(temperature))
        loss = float(-np.log(np.clip(scaled[np.arange(len(target)), target], 1e-9, 1)).mean())
        if loss < best_loss:
            best_temperature, best_loss = float(temperature), loss
    return round(best_temperature, 2)


def _candidate_cv(
    kind: str, x: list[str], y: list[str], folds: list[tuple[list[int], list[int]]]
) -> dict[str, Any]:
    predictions = np.empty(len(y), dtype=object)
    fit_seconds = 0.0
    prediction_seconds = 0.0
    fold_scores = []
    for train, validation in folds:
        model = _pipeline(kind)
        started = time.perf_counter()
        model.fit([x[index] for index in train], [y[index] for index in train])
        fit_seconds += time.perf_counter() - started
        started = time.perf_counter()
        fold_pred = model.predict([x[index] for index in validation])
        prediction_seconds += time.perf_counter() - started
        predictions[validation] = fold_pred
        fold_scores.append(f1_score([y[index] for index in validation], fold_pred, average="macro", zero_division=0))
    return {
        "model": kind, "grouped_cv_macro_f1": round(float(np.mean(fold_scores)), 4),
        "grouped_cv_std": round(float(np.std(fold_scores)), 4),
        "oof_accuracy": round(float(accuracy_score(y, predictions.tolist())), 4),
        "fit_seconds": round(fit_seconds, 3),
        "prediction_ms_per_ticket": round(1000 * prediction_seconds / len(y), 3),
        "offline": True, "external_cost": 0,
    }


def _oof_probabilities(kind: str, x: list[str], y: list[str], folds: list[tuple[list[int], list[int]]], labels: list[str]) -> np.ndarray:
    output = np.zeros((len(y), len(labels)))
    for train, validation in folds:
        model = _pipeline(kind)
        model.fit([x[index] for index in train], [y[index] for index in train])
        output[validation] = _class_probabilities(model, [x[index] for index in validation], labels)
    return output


def _choose_category_policy(
    probabilities: np.ndarray, y_true: list[str], labels: list[str], oos_probabilities: np.ndarray
) -> tuple[float, float, list[dict[str, float]]]:
    maximum = probabilities.max(axis=1)
    order = np.sort(probabilities, axis=1)
    margins = order[:, -1] - order[:, -2]
    predictions = np.asarray(labels)[probabilities.argmax(axis=1)]
    oos_maximum = oos_probabilities.max(axis=1)
    oos_order = np.sort(oos_probabilities, axis=1)
    oos_margins = oos_order[:, -1] - oos_order[:, -2]
    best = (0.45, 0.05, -1.0)
    curve: list[dict[str, float]] = []
    for threshold in np.arange(0.3, 0.76, 0.05):
        for margin in (0.0, 0.03, 0.06, 0.1):
            accepted = (maximum >= threshold) & (margins >= margin)
            oos_accepted = (oos_maximum >= threshold) & (oos_margins >= margin)
            coverage = float(accepted.mean())
            accuracy = float((predictions[accepted] == np.asarray(y_true)[accepted]).mean()) if accepted.any() else 0.0
            accepted_f1 = float(f1_score(np.asarray(y_true)[accepted], predictions[accepted], average="macro", zero_division=0)) if accepted.any() else 0.0
            rejection = float((~oos_accepted).mean()) if len(oos_accepted) else 0.0
            utility = 0.42 * accuracy + 0.30 * rejection + 0.18 * coverage + 0.10 * accepted_f1
            row = {"threshold": round(float(threshold), 2), "margin": margin, "coverage": round(coverage, 4),
                   "accepted_accuracy": round(accuracy, 4), "accepted_macro_f1": round(accepted_f1, 4),
                   "oos_rejection": round(rejection, 4), "oos_false_acceptance": round(1 - rejection, 4),
                   "utility": round(utility, 4)}
            curve.append(row)
            if utility > best[2]:
                best = (float(threshold), float(margin), utility)
    return round(best[0], 2), round(best[1], 2), curve


def _choose_routing_threshold(probabilities: np.ndarray, y_true: list[str], labels: list[str]) -> tuple[float, list[dict[str, float]]]:
    maximum = probabilities.max(axis=1)
    predictions = np.asarray(labels)[probabilities.argmax(axis=1)]
    curve = []
    best = (0.55, -1.0)
    operational_choice: float | None = None
    for threshold in np.arange(0.4, 0.81, 0.05):
        accepted = maximum >= threshold
        coverage = float(accepted.mean())
        accuracy = float((predictions[accepted] == np.asarray(y_true)[accepted]).mean()) if accepted.any() else 0.0
        utility = 0.65 * accuracy + 0.35 * coverage
        curve.append({"threshold": round(float(threshold), 2), "coverage": round(coverage, 4),
                      "accepted_accuracy": round(accuracy, 4), "utility": round(utility, 4)})
        if operational_choice is None and accuracy >= 0.82 and coverage >= 0.4:
            operational_choice = float(threshold)
        if utility > best[1]:
            best = (float(threshold), utility)
    return round(operational_choice if operational_choice is not None else max(0.55, best[0]), 2), curve


def duplicate_free_neighbors(scores: Sequence[float], query_index: int, groups: list[str], limit: int) -> list[int]:
    order = np.argsort(np.asarray(scores))[::-1]
    query_group = groups[query_index]
    return [int(index) for index in order if index != query_index and groups[int(index)] != query_group][:limit]


def _retrieval_vectorizer(kind: str) -> Any:
    if kind == "word":
        return _word_vectorizer(tuned=True)
    if kind == "char":
        return _char_vectorizer(tuned=True)
    return _word_char_features(tuned=True)


def _retrieval_scores(matrix: Any, rows: list[dict[str, Any]], index: int, metadata_boost: bool) -> np.ndarray:
    from sklearn.metrics.pairwise import cosine_similarity
    scores = cosine_similarity(matrix[index], matrix).ravel()
    if metadata_boost:
        query = rows[index]
        for candidate, row in enumerate(rows):
            scores[candidate] += 0.08 * bool(query["service"] and query["service"] == row["service"])
            scores[candidate] += 0.05 * bool(query["component"] and query["component"] == row["component"])
            scores[candidate] += 0.03 * bool(query["request_type"] and query["request_type"] == row["request_type"])
            scores[candidate] += 0.02 * bool(query["priority"] and query["priority"] == row["priority"])
    return scores


def _evaluate_retrieval(kind: str, matrix: Any, rows: list[dict[str, Any]], sample_indices: np.ndarray) -> dict[str, Any]:
    metadata_boost = kind.endswith("metadata")
    groups = [row["normalized_description"] or f"id:{row['request_id']}" for row in rows]
    hits = {1: 0, 3: 0, 5: 0}
    reciprocal_ranks = []
    examples = []
    started = time.perf_counter()
    for index in sample_indices:
        scores = _retrieval_scores(matrix, rows, int(index), metadata_boost)
        neighbors = duplicate_free_neighbors(scores, int(index), groups, 50)
        relevant_positions = [position + 1 for position, candidate in enumerate(neighbors) if rows[candidate]["category"] == rows[int(index)]["category"]]
        for k in hits:
            hits[k] += sum(rows[candidate]["category"] == rows[int(index)]["category"] for candidate in neighbors[:k])
        reciprocal_ranks.append(1 / relevant_positions[0] if relevant_positions else 0.0)
        if len(examples) < 12:
            top5 = neighbors[:5]
            examples.append({
                "query_request_id": rows[int(index)]["request_id"], "query_category": rows[int(index)]["category"],
                "top5_same_category": sum(rows[c]["category"] == rows[int(index)]["category"] for c in top5),
                "neighbors": [{"request_id": rows[c]["request_id"], "score": round(float(scores[c]), 4),
                               "category": rows[c]["category"]} for c in top5],
            })
    elapsed = time.perf_counter() - started
    query_count = len(sample_indices)
    return {
        "model": kind, "queries": query_count, "same_category_at_1": round(hits[1] / query_count, 4),
        "same_category_at_3": round(hits[3] / (query_count * 3), 4),
        "same_category_at_5": round(hits[5] / (query_count * 5), 4),
        "mrr_proxy": round(float(np.mean(reciprocal_ranks)), 4),
        "latency_ms_per_query": round(1000 * elapsed / query_count, 3), "duplicate_groups_excluded": True,
        "examples": examples,
    }


def _label_conflicts(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_description: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_features: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_description[row["normalized_description"]].append(row)
        by_features[registration_text(row)].append(row)
    def conflicts(groups: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
        result = []
        for key, group in groups.items():
            labels = sorted({row["category"] for row in group})
            if len(labels) > 1:
                result.append({"group_fingerprint": hashlib.sha256(key.encode()).hexdigest()[:12],
                               "request_ids": [row["request_id"] for row in group], "labels": labels, "count": len(group)})
        return sorted(result, key=lambda item: item["count"], reverse=True)
    description = conflicts(by_description)
    features = conflicts(by_features)
    return {
        "normalized_description_conflicts": {"groups": len(description), "records": sum(item["count"] for item in description), "examples": description[:50]},
        "registration_feature_conflicts": {"groups": len(features), "records": sum(item["count"] for item in features), "examples": features[:50]},
        "policy": "Исходные метки не исправлялись и конфликтующие записи не удалялись из оценки.",
    }


def _model_metadata(role: str, family: str, count: int, dataset_hash: str, feature_schema: list[str], split: str) -> dict[str, Any]:
    packages = {name: importlib.metadata.version(name) for name in ("scikit-learn", "numpy", "joblib")}
    serializable_parameters = {
        key: value
        for key, value in _pipeline(family).get_params(deep=True).items()
        if isinstance(value, (str, int, float, bool, type(None)))
    }
    return {"role": role, "model_family": family, "hyperparameters": serializable_parameters,
            "dataset_sha256": dataset_hash, "training_record_count": count, "trained_at": datetime.now(UTC).isoformat(),
            "package_versions": packages, "feature_schema": feature_schema, "split_methodology": split,
            "workspace_state": "no git repository; baseline artifacts preserved in artifacts/baseline_v1"}


def _save_bundle(path: Path, bundle: dict[str, Any], metadata_path: Path) -> None:
    previous = path.parent / "previous"
    previous.mkdir(exist_ok=True)
    baseline_path = previous / f"{path.stem}_v1_baseline.joblib"
    if path.exists() and not baseline_path.exists():
        shutil.copy2(path, baseline_path)
    joblib.dump(bundle, path)
    metadata = dict(bundle.get("metadata", {}))
    metadata["model_artifact_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    _write_json(metadata_path, metadata)


def train_all(database: Path, models_dir: Path, evaluation_dir: Path) -> dict[str, Any]:
    models_dir.mkdir(parents=True, exist_ok=True)
    evaluation_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = evaluation_dir.parent / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    rows = load_rows(database)
    summary = metadata_value("dataset_summary", database)
    dataset_hash = str(metadata_value("source_sha256", database) or "unknown")
    top15 = [item["name"] for item in summary["top15"]]
    feature_schema = ["description", "service", "component", "request_type", "criticality", "urgency", "priority", "service_class", "timezone"]

    in_scope = [row for row in rows if row["category"] in top15]
    oos = [row for row in rows if row["category"] not in top15]
    x = [registration_text(row) for row in in_scope]
    y = [row["category"] for row in in_scope]
    groups = [row["normalized_description"] or f"id:{row['request_id']}" for row in in_scope]
    split = make_canonical_split(y, groups)
    dev_x, dev_y = [x[index] for index in split.development], [y[index] for index in split.development]
    labels = sorted(set(y))

    candidates = [_candidate_cv(kind, dev_x, dev_y, split.cv_folds)
                  for kind in ("word_lr", "word_char_lr", "tuned_word_char_lr", "linear_svc")]
    best_linear = max(candidates, key=lambda item: item["grouped_cv_macro_f1"])
    probabilistic = [item for item in candidates if item["model"] != "linear_svc"]
    selected = max(probabilistic, key=lambda item: item["grouped_cv_macro_f1"])
    selection_reason = (
        "Выбрана лучшая вероятностная модель: преимущество LinearSVC недостаточно, чтобы отказаться от прозрачной калибровки."
        if best_linear["model"] == "linear_svc" else "Выбрана модель с максимальным grouped CV macro F1."
    )
    selected_kind = selected["model"]

    ablations = []
    for mode in ("text", "metadata", "combined"):
        values = [registration_text(row, mode) for row in in_scope]
        dev_values = [values[index] for index in split.development]
        result = _candidate_cv(selected_kind, dev_values, dev_y, split.cv_folds)
        ablations.append({"features": mode, "grouped_cv_macro_f1": result["grouped_cv_macro_f1"]})

    oof_raw = _oof_probabilities(selected_kind, dev_x, dev_y, split.cv_folds, labels)
    temperature = _fit_temperature(oof_raw, dev_y, labels)
    oof_calibrated = _temperature_scale(oof_raw, temperature)
    calibration = {
        "method": "temperature scaling по grouped OOF development predictions",
        "temperature": temperature,
        "brier_raw": round(_multiclass_brier(oof_raw, dev_y, labels), 4),
        "brier": round(_multiclass_brier(oof_calibrated, dev_y, labels), 4),
        "ece_raw": round(_ece(oof_raw, dev_y, labels), 4),
        "ece": round(_ece(oof_calibrated, dev_y, labels), 4),
    }

    oos_groups = [row["normalized_description"] or f"id:{row['request_id']}" for row in oos]
    oos_splitter = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=RANDOM_SEED)
    oos_validation_idx, oos_test_idx = next(oos_splitter.split(np.arange(len(oos)), [row["category"] for row in oos], oos_groups))
    policy_model = _pipeline(selected_kind)
    policy_model.fit(dev_x, dev_y)
    oos_validation_raw = _class_probabilities(policy_model, [registration_text(oos[index]) for index in oos_validation_idx], labels)
    threshold, margin, confidence_curve = _choose_category_policy(
        oof_calibrated, dev_y, labels, _temperature_scale(oos_validation_raw, temperature)
    )

    evaluation_estimator = _pipeline(selected_kind)
    evaluation_estimator.fit(dev_x, dev_y)
    evaluation_model = ProbabilityModel(evaluation_estimator, temperature)
    test_x, test_y = [x[index] for index in split.test], [y[index] for index in split.test]
    test_probabilities = evaluation_model.predict_proba(test_x)
    test_pred = evaluation_model.predict(test_x).tolist()
    category_metrics = _metrics(test_y, test_pred, labels)
    maximum = test_probabilities.max(axis=1)
    sorted_probabilities = np.sort(test_probabilities, axis=1)
    accepted = (maximum >= threshold) & ((sorted_probabilities[:, -1] - sorted_probabilities[:, -2]) >= margin)
    accepted_predictions = np.asarray(test_pred)[accepted]
    accepted_truth = np.asarray(test_y)[accepted]
    oos_test_raw = _class_probabilities(evaluation_estimator, [registration_text(oos[index]) for index in oos_test_idx], labels)
    oos_test = _temperature_scale(oos_test_raw, temperature)
    oos_sorted = np.sort(oos_test, axis=1)
    oos_accepted = (oos_test.max(axis=1) >= threshold) & ((oos_sorted[:, -1] - oos_sorted[:, -2]) >= margin)
    inference_started = time.perf_counter()
    evaluation_model.predict_proba(test_x[: min(100, len(test_x))])
    latency_ms = 1000 * (time.perf_counter() - inference_started) / min(100, len(test_x))
    category_metrics.update({
        "model": selected_kind, "selection_reason": selection_reason, "benchmark": candidates, "ablations": ablations,
        "confidence_threshold": threshold, "margin_threshold": margin, "accepted_coverage": round(float(accepted.mean()), 4),
        "accepted_accuracy": round(float((accepted_predictions == accepted_truth).mean()), 4) if accepted.any() else 0.0,
        "accepted_macro_f1": round(float(f1_score(accepted_truth, accepted_predictions, average="macro", zero_division=0)), 4) if accepted.any() else 0.0,
        "calibration": calibration, "latency_ms_per_ticket": round(latency_ms, 3),
        "split_strategy": "StratifiedGroupKFold: untouched grouped holdout 20%; 4-fold grouped CV on development",
        "out_of_scope": {"test_size": len(oos_test_idx), "rejection_rate": round(float((~oos_accepted).mean()), 4),
                         "false_acceptance_rate": round(float(oos_accepted.mean()), 4)},
    })

    deployment_estimator = _pipeline(selected_kind)
    deployment_estimator.fit(x, y)
    category_metadata = _model_metadata("deployment", selected_kind, len(in_scope), dataset_hash, feature_schema, category_metrics["split_strategy"])
    category_bundle = {"pipeline": ProbabilityModel(deployment_estimator, temperature), "explanation_pipeline": deployment_estimator,
                       "top15": top15, "threshold": threshold, "margin_threshold": margin, "metadata": category_metadata}
    _save_bundle(models_dir / "category.joblib", category_bundle, evaluation_dir / "category_model_metadata.json")

    routing_rows = [row for row in rows if row["final_line"] in {"(1 линия)", "(2 линия)", "(3 линия)"}]
    rx, ry = [registration_text(row) for row in routing_rows], [row["final_line"] for row in routing_rows]
    rgroups = [row["normalized_description"] or f"id:{row['request_id']}" for row in routing_rows]
    rsplit = make_canonical_split(ry, rgroups, RANDOM_SEED + 9)
    rdev_x, rdev_y = [rx[index] for index in rsplit.development], [ry[index] for index in rsplit.development]
    routing_candidates = [_candidate_cv(kind, rdev_x, rdev_y, rsplit.cv_folds)
                          for kind in ("word_char_lr", "tuned_word_char_lr", "linear_svc")]
    routing_selected = max([item for item in routing_candidates if item["model"] != "linear_svc"],
                           key=lambda item: item["grouped_cv_macro_f1"])
    routing_kind = routing_selected["model"]
    routing_labels = sorted(set(ry))
    routing_oof_raw = _oof_probabilities(routing_kind, rdev_x, rdev_y, rsplit.cv_folds, routing_labels)
    routing_temperature = _fit_temperature(routing_oof_raw, rdev_y, routing_labels)
    routing_oof = _temperature_scale(routing_oof_raw, routing_temperature)
    routing_threshold, routing_curve = _choose_routing_threshold(routing_oof, rdev_y, routing_labels)
    routing_eval_estimator = _pipeline(routing_kind)
    routing_eval_estimator.fit(rdev_x, rdev_y)
    routing_eval = ProbabilityModel(routing_eval_estimator, routing_temperature)
    routing_test_x, routing_test_y = [rx[index] for index in rsplit.test], [ry[index] for index in rsplit.test]
    routing_probabilities = routing_eval.predict_proba(routing_test_x)
    routing_pred = routing_eval.predict(routing_test_x).tolist()
    routing_metrics = _metrics(routing_test_y, routing_pred, routing_labels)
    routing_accepted = routing_probabilities.max(axis=1) >= routing_threshold
    routing_metrics.update({
        "model": routing_kind, "benchmark": routing_candidates, "confidence_threshold": routing_threshold,
        "accepted_coverage": round(float(routing_accepted.mean()), 4),
        "accepted_accuracy": round(float((np.asarray(routing_pred)[routing_accepted] == np.asarray(routing_test_y)[routing_accepted]).mean()), 4) if routing_accepted.any() else 0.0,
        "calibration": {"method": "temperature scaling по grouped OOF", "temperature": routing_temperature,
                        "brier": round(_multiclass_brier(_temperature_scale(routing_oof_raw, routing_temperature), rdev_y, routing_labels), 4),
                        "ece": round(_ece(_temperature_scale(routing_oof_raw, routing_temperature), rdev_y, routing_labels), 4)},
        "confidence_curve": routing_curve,
        "line4": "Исключена из ML-цели: один исторический пример; требуется ручная проверка.",
        "split_strategy": "StratifiedGroupKFold: untouched grouped holdout 20%; 4-fold grouped CV on development",
    })
    routing_deployment_estimator = _pipeline(routing_kind)
    routing_deployment_estimator.fit(rx, ry)
    routing_metadata = _model_metadata("deployment", routing_kind, len(routing_rows), dataset_hash, feature_schema, routing_metrics["split_strategy"])
    routing_bundle = {"pipeline": ProbabilityModel(routing_deployment_estimator, routing_temperature),
                      "explanation_pipeline": routing_deployment_estimator, "threshold": routing_threshold, "metadata": routing_metadata}
    _save_bundle(models_dir / "routing.joblib", routing_bundle, evaluation_dir / "routing_model_metadata.json")

    retrieval_x = [registration_text(row) for row in rows]
    sample_indices = np.linspace(0, len(rows) - 1, min(120, len(rows)), dtype=int)
    retrieval_candidates = []
    retrieval_artifacts: dict[str, tuple[Any, Any]] = {}
    for kind in ("word", "char", "word_char", "word_char_metadata"):
        base_kind = "word_char" if kind == "word_char_metadata" else kind
        vectorizer = _retrieval_vectorizer(base_kind)
        matrix = vectorizer.fit_transform(retrieval_x)
        retrieval_artifacts[kind] = (vectorizer, matrix)
        retrieval_candidates.append(_evaluate_retrieval(kind, matrix, rows, sample_indices))
    retrieval_winner = max(retrieval_candidates, key=lambda item: (item["same_category_at_5"], item["same_category_at_1"], item["mrr_proxy"]))
    winner_vectorizer, winner_matrix = retrieval_artifacts[retrieval_winner["model"]]
    retrieval_parameters = {key: value for key, value in winner_vectorizer.get_params(deep=True).items()
                            if isinstance(value, (str, int, float, bool, type(None)))}
    retrieval_metadata = {
        "role": "deployment", "model_family": retrieval_winner["model"],
        "hyperparameters": retrieval_parameters, "training_record_count": len(rows),
        "dataset_sha256": dataset_hash, "trained_at": datetime.now(UTC).isoformat(),
        "package_versions": {name: importlib.metadata.version(name) for name in ("scikit-learn", "numpy", "joblib")},
        "feature_schema": feature_schema,
        "split_methodology": "duplicate-free fixed-seed query sample; self and exact normalized-description group excluded",
        "workspace_state": "no git repository; baseline artifacts preserved in artifacts/baseline_v1",
        "duplicate_free_evaluation": True,
    }
    retrieval_bundle = {"vectorizer": winner_vectorizer, "matrix": winner_matrix, "rows": rows,
                        "metadata_boost": retrieval_winner["model"].endswith("metadata"),
                        "metadata": retrieval_metadata}
    _save_bundle(models_dir / "retrieval.joblib", retrieval_bundle, evaluation_dir / "retrieval_model_metadata.json")
    retrieval_metrics = {"method": retrieval_winner["model"],
                         "proxy": "same historical category after excluding self and exact normalized-description group",
                         **{key: value for key, value in retrieval_winner.items() if key not in {"model"}},
                         "benchmark": [{key: value for key, value in item.items() if key != "examples"} for item in retrieval_candidates]}
    review_path = evaluation_dir / "retrieval_review_template.csv"
    with review_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["query_request_id", "candidate_request_id", "query_category", "candidate_category", "score", "human_relevant", "notes"])
        writer.writeheader()
        for example in retrieval_winner["examples"][:10]:
            for neighbor in example["neighbors"][:4]:
                writer.writerow({"query_request_id": example["query_request_id"], "candidate_request_id": neighbor["request_id"],
                                 "query_category": example["query_category"], "candidate_category": neighbor["category"],
                                 "score": neighbor["score"], "human_relevant": "", "notes": ""})

    durations = [row["actual_duration_seconds"] for row in rows if row["actual_duration_seconds"] is not None]
    overdue = [row for row in rows if row["overdue"]]
    sla_analysis = {"authoritative_label": "Просрочен?*", "overdue_count": len(overdue),
                    "overdue_share": round(len(overdue) / len(rows), 6), "duration_available": len(durations),
                    "median_duration_seconds": int(np.median(durations)), "p90_duration_seconds": int(np.percentile(durations, 90)),
                    "risk_method": "Иерархические сегменты с minimum support и Beta smoothing; raw rate показывается отдельно.",
                    "limitation": "Точный производственный календарь и паузы не представлены; поле Просрочен?* сохраняется как источник истины."}
    limitations = {"status_graph": "В текущем наборе нет event history; страница Процесс показывает честный empty state и отдельное участие линий.",
                   "line4": "Один пример недостаточен для достоверной модели.", "sla": sla_analysis["limitation"],
                   "confidence": "Уверенность откалибрована temperature scaling на grouped OOF development predictions.",
                   "retrieval": "Proxy same-category не заменяет человеческую оценку; подготовлен пустой review template."}

    development_ids = [in_scope[index]["request_id"] for index in split.development]
    test_ids = [in_scope[index]["request_id"] for index in split.test]
    split_payload = {
        "seed": RANDOM_SEED, "strategy": category_metrics["split_strategy"],
        "category": {"development_request_ids": development_ids, "test_request_ids": test_ids,
                     "cv_folds": [{"train_request_ids": [in_scope[split.development[index]]["request_id"] for index in train],
                                   "validation_request_ids": [in_scope[split.development[index]]["request_id"] for index in validation]}
                                  for train, validation in split.cv_folds]},
        "duplicate_group": "lowercase + whitespace normalization of Описание 2",
        "audit": {"development_test_overlap": len(set(development_ids) & set(test_ids)), "all_disjoint": True},
    }
    confidence_payload = {"selected_threshold": threshold, "selected_margin": margin,
                          "calibration": {"brier": calibration["brier"], "ece": calibration["ece"], "temperature": temperature,
                                          "method": calibration["method"]}, "curve": confidence_curve}
    conflicts = _label_conflicts(rows)
    comparison = {"category": candidates, "category_ablations": ablations, "routing": routing_candidates,
                  "retrieval": [{key: value for key, value in item.items() if key != "examples"} for item in retrieval_candidates]}
    _write_json(evaluation_dir / "dataset_summary.json", summary)
    _write_json(evaluation_dir / "category_metrics.json", category_metrics)
    _write_json(evaluation_dir / "routing_metrics.json", routing_metrics)
    _write_json(evaluation_dir / "confidence_analysis.json", confidence_payload)
    _write_json(evaluation_dir / "retrieval_examples.json", retrieval_metrics)
    _write_json(evaluation_dir / "sla_analysis.json", sla_analysis)
    _write_json(evaluation_dir / "limitations.json", limitations)
    _write_json(evaluation_dir / "split_metadata.json", split_payload)
    _write_json(evaluation_dir / "label_conflicts.json", conflicts)
    _write_json(evaluation_dir / "model_comparison.json", comparison)
    _save_confusion(category_metrics, figures_dir / "category_confusion_matrix.png", "Матрица ошибок: категории")
    _save_confusion(routing_metrics, figures_dir / "routing_confusion_matrix.png", "Матрица ошибок: линии поддержки")
    return {"category": category_metrics, "routing": routing_metrics, "retrieval": retrieval_metrics, "sla": sla_analysis}
