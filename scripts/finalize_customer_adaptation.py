from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import joblib
import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import LinearSVC

ROOT = Path(__file__).resolve().parents[1]
for import_path in (ROOT, ROOT / "backend"):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from app.ml.lite_v3_runtime import LiteV3DistilledCategoryPipeline  # noqa: E402
from app.ml.training import ProbabilityModel, _pipeline, registration_text  # noqa: E402
from app.ml.v5_runtime import (  # noqa: E402
    DecisionSoftmaxClassifier,
    QwenV5CategoryPipeline,
    build_v5_structured_features,
    specialist_text,
)
from scripts.lite_v3_local_max import MINILM_ID, load_qwen_teacher_embeddings  # noqa: E402
from scripts.v5_dataset import DEFAULT_DATASET, load_v5_rows  # noqa: E402
from scripts.v5_experiment import EXTENDED_INSTRUCTION_REGISTRY  # noqa: E402


SEED = 20260921
EVAL_DIR = ROOT / "outputs/customer_eval_2026-09-21"
NEW_ROWS_PATH = EVAL_DIR / "new_rows.json"
NEW_MINILM_PATH = EVAL_DIR / "minilm_new.npy"
NEW_QWEN_PATH = EVAL_DIR / "qwen4b_new.npy"
OLD_MINILM_PATH = ROOT / "outputs/lite_v3/minilm_256.npy"
BACKUP_QWEN_PATH = ROOT / "backups/customer_adaptation_2026-09-21/category_qwen4b_lite.joblib"
FOLDS_PATH = ROOT / "artifacts/gpu_research_v5/protocol/folds.json"
CUSTOMER_SOURCE = ROOT / "data/raw/ALL_TICKETS.xlsx"

MODEL_ID = "Qwen/Qwen3-Embedding-4B"
MODEL_REVISION = "5cf2132abc99cad020ac570b19d031efec650f2b"
INSTRUCTION_KEY = "posttech_tight_c"
EMBEDDING_DIM = 2560
MAX_LENGTH = 512
QWEN_METADATA_SCALE = 0.75
LITE_METADATA_SCALE = 0.35
KNN_NEIGHBORS = 11
PROTOTYPE_TEMPERATURE = 0.08


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize(values: np.ndarray) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return np.divide(matrix, norms, out=np.zeros_like(matrix), where=norms > 0)


def softmax(values: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    scores = np.asarray(values, dtype=float)
    if scores.ndim == 1:
        scores = np.column_stack([-scores, scores])
    scores = scores / max(float(temperature), 1e-6)
    scores -= scores.max(axis=1, keepdims=True)
    exp = np.exp(scores)
    return exp / exp.sum(axis=1, keepdims=True)


def aligned(values: np.ndarray, classes: Sequence[str], labels: Sequence[str]) -> np.ndarray:
    result = np.zeros((len(values), len(labels)), dtype=float)
    positions = {str(label): index for index, label in enumerate(labels)}
    for source, label in enumerate(classes):
        target = positions.get(str(label))
        if target is not None:
            result[:, target] = values[:, source]
    sums = result.sum(axis=1, keepdims=True)
    return np.divide(result, sums, out=np.zeros_like(result), where=sums > 0)


def topk_metrics(y_true: Sequence[str], probabilities: np.ndarray, labels: Sequence[str]) -> dict[str, float]:
    labels = [str(value) for value in labels]
    truth = np.asarray([str(value) for value in y_true], dtype=object)
    order = np.argsort(-probabilities, axis=1)
    pred = np.asarray([labels[int(index)] for index in order[:, 0]], dtype=object)
    top3 = [[labels[int(index)] for index in row[:3]] for row in order]
    present = sorted(set(truth.tolist()))
    return {
        "top1": float(accuracy_score(truth, pred)),
        "top3": float(np.mean([target in candidates for target, candidates in zip(truth, top3, strict=True)])),
        "macro_f1_present_truth": float(f1_score(truth, pred, labels=present, average="macro", zero_division=0)),
    }


def load_new_rows() -> list[dict[str, Any]]:
    raw = json.loads(NEW_ROWS_PATH.read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    for source in raw:
        row = dict(source)
        row["category"] = str(row.pop("truth_category"))
        row["routing_target"] = str(row.pop("truth_route"))
        rows.append(row)
    return rows


def load_all() -> tuple[list[dict[str, Any]], list[dict[str, Any]], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    old_rows = load_v5_rows(DEFAULT_DATASET)
    new_rows = load_new_rows()
    old_minilm = normalize(np.load(OLD_MINILM_PATH))
    new_minilm = normalize(np.load(NEW_MINILM_PATH))
    old_qwen = normalize(load_qwen_teacher_embeddings(BACKUP_QWEN_PATH, old_rows))
    new_qwen = normalize(np.load(NEW_QWEN_PATH))
    if not (len(old_rows) == len(old_minilm) == len(old_qwen) == 1931):
        raise RuntimeError("Old-row embedding alignment failed")
    if not (len(new_rows) == len(new_minilm) == len(new_qwen) == 66):
        raise RuntimeError("Customer-row embedding alignment failed")
    old_ids = {str(row["request_id"]) for row in old_rows}
    new_ids = {str(row["request_id"]) for row in new_rows}
    if old_ids & new_ids:
        raise RuntimeError("Old and customer request ids overlap")
    return old_rows, new_rows, old_minilm, new_minilm, old_qwen, new_qwen


def weighted_centroids(
    embeddings: np.ndarray,
    targets: Sequence[str],
    weights: np.ndarray,
    labels: Sequence[str],
) -> np.ndarray:
    embeddings = normalize(embeddings)
    targets_arr = np.asarray(targets, dtype=object)
    centroids = []
    for label in labels:
        mask = targets_arr == str(label)
        if not mask.any():
            centroids.append(np.zeros(embeddings.shape[1], dtype=np.float32))
            continue
        centroid = np.average(embeddings[mask], axis=0, weights=weights[mask])
        centroids.append(normalize(np.asarray(centroid, dtype=np.float32)[None, :])[0])
    return np.asarray(centroids, dtype=np.float32)


def repeated_knn_training(
    embeddings: np.ndarray,
    targets: Sequence[str],
    weights: np.ndarray,
) -> tuple[np.ndarray, list[str]]:
    repeats = np.maximum(1, np.rint(weights).astype(int))
    matrix = np.repeat(embeddings, repeats, axis=0)
    labels: list[str] = []
    for target, count in zip(targets, repeats, strict=True):
        labels.extend([str(target)] * int(count))
    return matrix, labels


def qwen_fold_predict(
    train_rows: Sequence[Mapping[str, Any]],
    train_embeddings: np.ndarray,
    train_targets: Sequence[str],
    train_weights: np.ndarray,
    validation_rows: Sequence[Mapping[str, Any]],
    validation_embeddings: np.ndarray,
    labels: Sequence[str],
) -> np.ndarray:
    metadata_encoder = OneHotEncoder(handle_unknown="ignore", min_frequency=2, sparse_output=False)
    train_meta = np.asarray(metadata_encoder.fit_transform(build_v5_structured_features(train_rows)), dtype=float)
    valid_meta = np.asarray(metadata_encoder.transform(build_v5_structured_features(validation_rows)), dtype=float)
    train_x = np.concatenate([train_embeddings, train_meta * QWEN_METADATA_SCALE], axis=1)
    valid_x = np.concatenate([validation_embeddings, valid_meta * QWEN_METADATA_SCALE], axis=1)
    classifier = LinearSVC(C=1.0, class_weight="balanced", random_state=SEED, max_iter=10000)
    classifier.fit(train_x, list(map(str, train_targets)), sample_weight=train_weights)
    supervised = aligned(softmax(classifier.decision_function(valid_x)), classifier.classes_, labels)

    centroids = weighted_centroids(train_embeddings, train_targets, train_weights, labels)
    prototype = softmax(normalize(validation_embeddings) @ centroids.T, PROTOTYPE_TEMPERATURE)

    knn_x, knn_y = repeated_knn_training(train_embeddings, train_targets, train_weights)
    knn = KNeighborsClassifier(
        n_neighbors=max(1, min(KNN_NEIGHBORS, len(knn_y))),
        weights="distance",
        metric="cosine",
        algorithm="brute",
    ).fit(knn_x, knn_y)
    knn_p = aligned(knn.predict_proba(validation_embeddings), knn.classes_, labels)
    result = 0.90 * supervised + 0.05 * prototype + 0.05 * knn_p
    return result / result.sum(axis=1, keepdims=True)


def lite_text_features() -> FeatureUnion:
    return FeatureUnion([
        (
            "word",
            TfidfVectorizer(
                ngram_range=(1, 1),
                min_df=2,
                max_df=0.995,
                max_features=30000,
                sublinear_tf=True,
            ),
        ),
        (
            "char",
            TfidfVectorizer(
                analyzer="char_wb",
                ngram_range=(3, 5),
                min_df=2,
                max_features=60000,
                sublinear_tf=True,
            ),
        ),
    ])


def lite_fold_predict(
    train_rows: Sequence[Mapping[str, Any]],
    train_minilm: np.ndarray,
    train_qwen: np.ndarray,
    train_targets: Sequence[str],
    train_weights: np.ndarray,
    validation_rows: Sequence[Mapping[str, Any]],
    validation_minilm: np.ndarray,
    labels: Sequence[str],
) -> np.ndarray:
    projector = Ridge(alpha=1.0, fit_intercept=False)
    projector.fit(train_minilm, train_qwen, sample_weight=train_weights)
    projected_train = normalize(projector.predict(train_minilm))
    projected_valid = normalize(projector.predict(validation_minilm))
    vectorizer = lite_text_features()
    text_train = vectorizer.fit_transform([specialist_text(row) for row in train_rows])
    text_valid = vectorizer.transform([specialist_text(row) for row in validation_rows])
    metadata_encoder = OneHotEncoder(handle_unknown="ignore", min_frequency=2, sparse_output=True)
    meta_train = metadata_encoder.fit_transform(build_v5_structured_features(train_rows))
    meta_valid = metadata_encoder.transform(build_v5_structured_features(validation_rows))
    x_train = sparse.hstack(
        [text_train, meta_train * LITE_METADATA_SCALE, sparse.csr_matrix(projected_train)],
        format="csr",
    )
    x_valid = sparse.hstack(
        [text_valid, meta_valid * LITE_METADATA_SCALE, sparse.csr_matrix(projected_valid)],
        format="csr",
    )
    classifier = LinearSVC(C=0.5, class_weight="balanced", max_iter=10000, random_state=SEED)
    classifier.fit(x_train, list(map(str, train_targets)), sample_weight=train_weights)
    return aligned(softmax(classifier.decision_function(x_valid), 0.173915), classifier.classes_, labels)


def evaluate_old_development(
    old_rows: list[dict[str, Any]],
    new_rows: list[dict[str, Any]],
    old_minilm: np.ndarray,
    new_minilm: np.ndarray,
    old_qwen: np.ndarray,
    new_qwen: np.ndarray,
) -> dict[str, Any]:
    folds = json.loads(FOLDS_PATH.read_text(encoding="utf-8"))
    old_index = {str(row["request_id"]): index for index, row in enumerate(old_rows)}
    labels = sorted({str(row["category"]) for row in old_rows + new_rows})
    results: dict[str, list[dict[str, float]]] = {
        "qwen_w8": [],
        "qwen_w16": [],
        "lite_w4": [],
        "lite_w8": [],
    }
    for fold in folds:
        train_idx = [old_index[str(value)] for value in fold["train_request_ids"]]
        valid_idx = [old_index[str(value)] for value in fold["validation_request_ids"]]
        train_old = [old_rows[index] for index in train_idx]
        valid_rows = [old_rows[index] for index in valid_idx]
        train_targets_base = [str(row["category"]) for row in train_old]
        new_targets = [str(row["category"]) for row in new_rows]
        targets = train_targets_base + new_targets
        qwen_train = np.vstack([old_qwen[train_idx], new_qwen])
        minilm_train = np.vstack([old_minilm[train_idx], new_minilm])
        qwen_teacher_train = qwen_train
        train_rows = train_old + new_rows
        y_valid = [str(row["category"]) for row in valid_rows]

        for weight in (8.0, 16.0):
            sample_weights = np.concatenate([np.ones(len(train_old)), np.full(len(new_rows), weight)])
            probabilities = qwen_fold_predict(
                train_rows,
                qwen_train,
                targets,
                sample_weights,
                valid_rows,
                old_qwen[valid_idx],
                labels,
            )
            results[f"qwen_w{int(weight)}"].append(topk_metrics(y_valid, probabilities, labels))

        for weight in (4.0, 8.0):
            sample_weights = np.concatenate([np.ones(len(train_old)), np.full(len(new_rows), weight)])
            probabilities = lite_fold_predict(
                train_rows,
                minilm_train,
                qwen_teacher_train,
                targets,
                sample_weights,
                valid_rows,
                old_minilm[valid_idx],
                labels,
            )
            results[f"lite_w{int(weight)}"].append(topk_metrics(y_valid, probabilities, labels))
        print(f"old-development fold {fold['repeat']}/{fold['fold']} complete", flush=True)

    summary: dict[str, Any] = {}
    for name, rows in results.items():
        summary[name] = {
            key: float(np.mean([row[key] for row in rows]))
            for key in ("top1", "top3", "macro_f1_present_truth")
        }
    return {
        "method": "12 frozen V5 DEVELOPMENT folds; customer 66 always train-only; no lockbox access",
        "labels_union": len(labels),
        "summary": summary,
        "folds": results,
    }


def routing_metrics_current(new_rows: list[dict[str, Any]]) -> dict[str, float]:
    bundle = joblib.load(ROOT / "models/routing.joblib")
    texts = [registration_text(row) for row in new_rows]
    truth = [str(row["routing_target"]) for row in new_rows]
    pred = [str(value) for value in bundle["pipeline"].predict(texts)]
    labels = sorted(set(truth))
    return {
        "accuracy": float(accuracy_score(truth, pred)),
        "macro_f1": float(f1_score(truth, pred, labels=labels, average="macro", zero_division=0)),
    }


def customer_route_folds(new_rows: list[dict[str, Any]]) -> list[tuple[list[int], list[int]]]:
    buckets: dict[str, list[int]] = {}
    for index, row in enumerate(new_rows):
        buckets.setdefault(str(row["routing_target"]), []).append(index)
    folds = [[], []]
    for values in buckets.values():
        for offset, index in enumerate(values):
            folds[offset % 2].append(index)
    all_indices = set(range(len(new_rows)))
    return [(sorted(all_indices - set(valid)), sorted(valid)) for valid in folds]


def evaluate_routing_adaptation(old_rows: list[dict[str, Any]], new_rows: list[dict[str, Any]]) -> dict[str, Any]:
    old_route_rows = [
        row for row in old_rows
        if str(row["routing_target"]) in {"(1 линия)", "(2 линия)", "(3 линия)"}
    ]
    configs = [
        ("word_char_lr", 2.0),
        ("word_char_lr", 4.0),
        ("tuned_word_char_lr", 2.0),
        ("tuned_word_char_lr", 4.0),
    ]
    folds = customer_route_folds(new_rows)
    results = []
    for kind, customer_weight in configs:
        truths: list[str] = []
        predictions: list[str] = []
        for train_new_idx, valid_new_idx in folds:
            train_rows = old_route_rows + [new_rows[index] for index in train_new_idx]
            valid_rows = [new_rows[index] for index in valid_new_idx]
            x_train = [registration_text(row) for row in train_rows]
            y_train = [str(row["routing_target"]) for row in train_rows]
            weights = np.concatenate([
                np.ones(len(old_route_rows)),
                np.full(len(train_new_idx), customer_weight),
            ])
            model = _pipeline(kind)
            model.fit(x_train, y_train, classifier__sample_weight=weights)
            predictions.extend([str(value) for value in model.predict([registration_text(row) for row in valid_rows])])
            truths.extend([str(row["routing_target"]) for row in valid_rows])
        labels = sorted(set(truths))
        results.append({
            "kind": kind,
            "customer_weight": customer_weight,
            "accuracy": float(accuracy_score(truths, predictions)),
            "macro_f1": float(f1_score(truths, predictions, labels=labels, average="macro", zero_division=0)),
        })
    winner = max(results, key=lambda row: (row["accuracy"], row["macro_f1"]))
    return {
        "current_external": routing_metrics_current(new_rows),
        "method": "2-fold customer OOF; historical routing rows always train-only",
        "results": results,
        "winner": winner,
    }


def save_combined_manifest(old_rows: list[dict[str, Any]], new_rows: list[dict[str, Any]]) -> dict[str, Any]:
    processed = ROOT / "data/processed"
    processed.mkdir(parents=True, exist_ok=True)
    combined_path = processed / "customer_adapted_1997.json"
    canonical = old_rows + new_rows
    combined_path.write_text(json.dumps(canonical, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest = {
        "created_at": datetime.now(UTC).isoformat(),
        "old_rows": len(old_rows),
        "customer_rows": len(new_rows),
        "combined_rows": len(canonical),
        "category_labels": len({str(row["category"]) for row in canonical}),
        "routing_labels": sorted({str(row["routing_target"]) for row in canonical}),
        "old_dataset": str(DEFAULT_DATASET),
        "old_dataset_sha256": sha256(DEFAULT_DATASET),
        "customer_dataset": str(CUSTOMER_SOURCE),
        "customer_dataset_sha256": sha256(CUSTOMER_SOURCE) if CUSTOMER_SOURCE.exists() else None,
        "combined_json": str(combined_path),
        "combined_json_sha256": sha256(combined_path),
        "new_category_labels": sorted(
            {str(row["category"]) for row in new_rows} - {str(row["category"]) for row in old_rows}
        ),
    }
    manifest_path = processed / "customer_adapted_1997_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def build_lite(
    old_rows: list[dict[str, Any]],
    new_rows: list[dict[str, Any]],
    old_minilm: np.ndarray,
    new_minilm: np.ndarray,
    old_qwen: np.ndarray,
    new_qwen: np.ndarray,
    manifest: Mapping[str, Any],
    validation: Mapping[str, Any],
    customer_weight: float,
) -> dict[str, Any]:
    rows = old_rows + new_rows
    labels = sorted({str(row["category"]) for row in rows})
    targets = [str(row["category"]) for row in rows]
    minilm = np.vstack([old_minilm, new_minilm])
    qwen = np.vstack([old_qwen, new_qwen])
    weights = np.concatenate([np.ones(len(old_rows)), np.full(len(new_rows), customer_weight)])
    projector = Ridge(alpha=1.0, fit_intercept=False)
    projector.fit(minilm, qwen, sample_weight=weights)
    projected = normalize(projector.predict(minilm))

    vectorizer = lite_text_features()
    text_matrix = vectorizer.fit_transform([specialist_text(row) for row in rows])
    metadata_encoder = OneHotEncoder(handle_unknown="ignore", min_frequency=2, sparse_output=True)
    metadata_matrix = metadata_encoder.fit_transform(build_v5_structured_features(rows))
    matrix = sparse.hstack(
        [text_matrix, metadata_matrix * LITE_METADATA_SCALE, sparse.csr_matrix(projected)],
        format="csr",
    )
    classifier = LinearSVC(C=0.5, class_weight="balanced", max_iter=10000, random_state=SEED)
    classifier.fit(matrix, targets, sample_weight=weights)

    old_bundle = joblib.load(ROOT / "backups/customer_adaptation_2026-09-21/category_lite_v3.joblib")
    old_pipeline = old_bundle["pipeline"]
    specialist_vectorizer = FeatureUnion([
        (
            "char",
            TfidfVectorizer(
                analyzer="char_wb",
                ngram_range=(3, 5),
                min_df=2,
                max_features=60000,
                sublinear_tf=True,
            ),
        ),
        (
            "word",
            TfidfVectorizer(
                ngram_range=(1, 2),
                min_df=2,
                max_features=30000,
                sublinear_tf=True,
            ),
        ),
    ])
    specialist_matrix = specialist_vectorizer.fit_transform([specialist_text(row) for row in rows])
    specialist_model = LinearSVC(C=1.0, class_weight="balanced", max_iter=10000, random_state=SEED)
    specialist_model.fit(specialist_matrix, targets, sample_weight=weights)

    pipeline = LiteV3DistilledCategoryPipeline(
        labels=labels,
        vectorizer=vectorizer,
        metadata_encoder=metadata_encoder,
        classifier=classifier,
        metadata_scale=LITE_METADATA_SCALE,
        temperature=0.173915,
        text_mode="specialist",
        projector=projector,
        minilm_model_id=MINILM_ID,
        local_model_dir="models/encoders/paraphrase-multilingual-MiniLM-L12-v2",
        max_seq_length=256,
        batch_size=16,
        specialist_vectorizer=specialist_vectorizer,
        specialist_model=specialist_model,
        specialist_pairs=[list(pair) for pair in getattr(old_pipeline, "specialist_pairs", set())],
        specialist_threshold=float(getattr(old_pipeline, "specialist_threshold", 0.0)),
    )
    metadata = {
        "version": "v5.5-lite-v3-customer-adapted",
        "candidate_id": "lite-v3-qdistill-customer49-c05-w4",
        "profile": "lite",
        "model_family": "lite_v3_qwen_distilled_customer_adapted",
        "family": "TF-IDF word+char + metadata + Qwen-distilled MiniLM projection + LinearSVC + validated legacy specialist",
        "trained_at": datetime.now(UTC).isoformat(),
        "training_record_count": len(rows),
        "historical_training_rows": len(old_rows),
        "customer_training_rows": len(new_rows),
        "labels": len(labels),
        "label_names": labels,
        "dataset_sha256": manifest["combined_json_sha256"],
        "source_manifest": "data/processed/customer_adapted_1997_manifest.json",
        "input_contract": "registration_mapping",
        "student_model": MINILM_ID,
        "student_local_dir": "models/encoders/paraphrase-multilingual-MiniLM-L12-v2",
        "teacher": "saved Qwen3-Embedding-4B 2560-d embeddings; Qwen not required at Lite runtime",
        "projection": {"type": "Ridge", "alpha": 1.0, "fit_intercept": False, "output_dim": 2560},
        "classifier": {"type": "LinearSVC", "c": 0.5, "class_weight": "balanced", "customer_weight": customer_weight},
        "customer_external_oof": json.loads((EVAL_DIR / "lite_adaptation_sweep.json").read_text(encoding="utf-8"))["winner"],
        "historical_development_stability": validation,
        "lockbox_accessed": False,
    }
    bundle = {
        "pipeline": pipeline,
        "top15": old_bundle.get("top15", []),
        "threshold": 0.36,
        "margin_threshold": 0.0,
        "input_contract": "registration_mapping",
        "metadata": metadata,
    }
    output = ROOT / "models/v5/category_lite_v3.joblib"
    joblib.dump(bundle, output, compress=3)
    metadata["artifact_sha256"] = sha256(output)
    metadata["artifact_bytes"] = output.stat().st_size
    output.with_suffix(".json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return metadata


def build_qwen(
    old_rows: list[dict[str, Any]],
    new_rows: list[dict[str, Any]],
    old_qwen: np.ndarray,
    new_qwen: np.ndarray,
    manifest: Mapping[str, Any],
    validation: Mapping[str, Any],
    customer_weight: float,
) -> dict[str, Any]:
    rows = old_rows + new_rows
    embeddings = normalize(np.vstack([old_qwen, new_qwen]))
    targets = [str(row["category"]) for row in rows]
    labels = sorted(set(targets))
    weights = np.concatenate([np.ones(len(old_rows)), np.full(len(new_rows), customer_weight)])
    metadata_encoder = OneHotEncoder(handle_unknown="ignore", min_frequency=2, sparse_output=False)
    metadata_matrix = np.asarray(metadata_encoder.fit_transform(build_v5_structured_features(rows)), dtype=float)
    supervised_x = np.concatenate([embeddings, metadata_matrix * QWEN_METADATA_SCALE], axis=1)
    estimator = LinearSVC(C=1.0, class_weight="balanced", random_state=SEED, max_iter=10000)
    estimator.fit(supervised_x, targets, sample_weight=weights)
    supervised = DecisionSoftmaxClassifier(estimator, temperature=1.0)
    centroids = weighted_centroids(embeddings, targets, weights, labels)
    knn_x, knn_y = repeated_knn_training(embeddings, targets, weights)
    knn = KNeighborsClassifier(
        n_neighbors=max(1, min(KNN_NEIGHBORS, len(knn_y))),
        weights="distance",
        metric="cosine",
        algorithm="brute",
    ).fit(knn_x, knn_y)

    pipeline = QwenV5CategoryPipeline(
        labels=labels,
        model_id=MODEL_ID,
        model_revision=MODEL_REVISION,
        instruction=EXTENDED_INSTRUCTION_REGISTRY[INSTRUCTION_KEY],
        max_length=MAX_LENGTH,
        embedding_dim=EMBEDDING_DIM,
        metadata_encoder=metadata_encoder,
        supervised_model=supervised,
        prototype_centroids=centroids,
        knn_model=knn,
        metadata_scale=QWEN_METADATA_SCALE,
        prototype_temperature=PROTOTYPE_TEMPERATURE,
        batch_size=8,
    )
    old_bundle = joblib.load(BACKUP_QWEN_PATH)
    metadata = {
        "version": "v5.3-qwen4b-lite-customer-adapted",
        "profile": "manual_recheck",
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "view": "customer49",
        "instruction": INSTRUCTION_KEY,
        "embedding_dim": EMBEDDING_DIM,
        "max_length": MAX_LENGTH,
        "metadata_scale": QWEN_METADATA_SCALE,
        "supervised_head": "linearsvc_c1_decision_softmax",
        "customer_weight": customer_weight,
        "blend": {"supervised": 0.90, "prototype": 0.05, "knn": 0.05},
        "knn_neighbors": KNN_NEIGHBORS,
        "prototype_temperature": PROTOTYPE_TEMPERATURE,
        "real_training_rows": len(rows),
        "historical_training_rows": len(old_rows),
        "customer_training_rows": len(new_rows),
        "labels": len(labels),
        "label_names": labels,
        "dataset_sha256": manifest["combined_json_sha256"],
        "source_manifest": "data/processed/customer_adapted_1997_manifest.json",
        "customer_external_oof": json.loads((EVAL_DIR / "qwen4b_adaptation_sweep.json").read_text(encoding="utf-8"))["winner"],
        "historical_development_stability": validation,
        "previous_development_metrics": old_bundle.get("metadata", {}).get("development_metrics", {}),
        "lockbox_accessed": False,
    }
    bundle = {
        "pipeline": pipeline,
        "top15": old_bundle.get("top15", []),
        "threshold": 0.0,
        "margin_threshold": 0.0,
        "input_contract": "registration_mapping",
        "metadata": metadata,
    }
    output = ROOT / "models/v5/category_qwen4b_lite.joblib"
    joblib.dump(bundle, output, compress=3)
    metadata["artifact_sha256"] = sha256(output)
    metadata["artifact_bytes"] = output.stat().st_size
    output.with_suffix(".json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return metadata


def build_routing(
    old_rows: list[dict[str, Any]],
    new_rows: list[dict[str, Any]],
    route_eval: Mapping[str, Any],
) -> dict[str, Any]:
    winner = dict(route_eval["winner"])
    current = dict(route_eval["current_external"])
    if winner["accuracy"] + 1e-12 < current["accuracy"]:
        return {"retrained": False, "reason": "OOF adaptation did not beat current external accuracy", **current}
    old_route_rows = [
        row for row in old_rows
        if str(row["routing_target"]) in {"(1 линия)", "(2 линия)", "(3 линия)"}
    ]
    rows = old_route_rows + new_rows
    x = [registration_text(row) for row in rows]
    y = [str(row["routing_target"]) for row in rows]
    customer_weight = float(winner["customer_weight"])
    weights = np.concatenate([np.ones(len(old_route_rows)), np.full(len(new_rows), customer_weight)])
    model = _pipeline(str(winner["kind"]))
    model.fit(x, y, classifier__sample_weight=weights)
    previous = joblib.load(ROOT / "backups/customer_adaptation_2026-09-21/routing.joblib")
    bundle = {
        "pipeline": ProbabilityModel(model, 1.0),
        "explanation_pipeline": model,
        "threshold": float(previous.get("threshold", 0.55)),
        "metadata": {
            "version": "customer-adapted-2026-09-21",
            "model_family": str(winner["kind"]),
            "training_record_count": len(rows),
            "historical_training_rows": len(old_route_rows),
            "customer_training_rows": len(new_rows),
            "customer_weight": customer_weight,
            "customer_external_before": current,
            "customer_oof_after": winner,
            "trained_at": datetime.now(UTC).isoformat(),
        },
    }
    joblib.dump(bundle, ROOT / "models/routing.joblib", compress=3)
    return {"retrained": True, **bundle["metadata"], "artifact_sha256": sha256(ROOT / "models/routing.joblib")}


def in_sample_smoke(
    old_rows: list[dict[str, Any]],
    new_rows: list[dict[str, Any]],
    old_qwen: np.ndarray,
    new_qwen: np.ndarray,
) -> dict[str, Any]:
    qwen_bundle = joblib.load(ROOT / "models/v5/category_qwen4b_lite.joblib")
    qwen_prob = qwen_bundle["pipeline"].predict_proba_from_embeddings(new_rows, new_qwen)
    qwen_metrics = topk_metrics([row["category"] for row in new_rows], qwen_prob, qwen_bundle["pipeline"].labels)
    lite_bundle = joblib.load(ROOT / "models/v5/category_lite_v3.joblib")
    # This smoke intentionally uses the fitted feature heads with cached MiniLM values
    # rather than loading MiniLM a second time. It is only an in-sample artifact sanity check.
    lite_pipeline = lite_bundle["pipeline"]
    all_minilm = np.vstack([np.load(OLD_MINILM_PATH), np.load(NEW_MINILM_PATH)])
    projected = normalize(lite_pipeline.projector.predict(all_minilm[-len(new_rows):]))
    text = lite_pipeline.vectorizer.transform([specialist_text(row) for row in new_rows])
    meta = lite_pipeline.metadata_encoder.transform(build_v5_structured_features(new_rows))
    matrix = sparse.hstack([text, meta * lite_pipeline.metadata_scale, sparse.csr_matrix(projected)], format="csr")
    local = softmax(lite_pipeline.classifier.decision_function(matrix), lite_pipeline.temperature)
    lite_prob = aligned(local, lite_pipeline.classifier.classes_, lite_pipeline.labels)
    lite_prob = lite_pipeline._apply_specialist(new_rows, lite_prob)
    lite_metrics = topk_metrics([row["category"] for row in new_rows], lite_prob, lite_pipeline.labels)
    return {"note": "in-sample smoke only; not external accuracy", "lite": lite_metrics, "qwen": qwen_metrics}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluate-only", action="store_true")
    parser.add_argument("--build", action="store_true")
    args = parser.parse_args()

    old_rows, new_rows, old_minilm, new_minilm, old_qwen, new_qwen = load_all()
    manifest = save_combined_manifest(old_rows, new_rows)
    stability_path = EVAL_DIR / "historical_development_stability.json"
    if stability_path.exists():
        stability = json.loads(stability_path.read_text(encoding="utf-8"))
    else:
        stability = evaluate_old_development(old_rows, new_rows, old_minilm, new_minilm, old_qwen, new_qwen)
        stability_path.write_text(json.dumps(stability, ensure_ascii=False, indent=2), encoding="utf-8")
    route_eval_path = EVAL_DIR / "routing_adaptation_eval.json"
    route_eval = evaluate_routing_adaptation(old_rows, new_rows)
    route_eval_path.write_text(json.dumps(route_eval, ensure_ascii=False, indent=2), encoding="utf-8")

    payload: dict[str, Any] = {"manifest": manifest, "historical_stability": stability, "routing": route_eval}
    if args.evaluate_only and not args.build:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    if args.build:
        qwen_summary = stability["summary"]
        # New-customer OOF gives identical TOP3 for w8/w16 and higher TOP1 for w16.
        # Prefer w16 unless it loses >1.5 percentage points old-development TOP1 vs w8.
        qwen_weight = 16.0
        if qwen_summary["qwen_w16"]["top1"] + 0.015 < qwen_summary["qwen_w8"]["top1"]:
            qwen_weight = 8.0
        lite_weight = 4.0
        lite_metadata = build_lite(
            old_rows,
            new_rows,
            old_minilm,
            new_minilm,
            old_qwen,
            new_qwen,
            manifest,
            qwen_summary[f"lite_w{int(lite_weight)}"],
            lite_weight,
        )
        qwen_metadata = build_qwen(
            old_rows,
            new_rows,
            old_qwen,
            new_qwen,
            manifest,
            qwen_summary[f"qwen_w{int(qwen_weight)}"],
            qwen_weight,
        )
        routing = build_routing(old_rows, new_rows, route_eval)
        smoke = in_sample_smoke(old_rows, new_rows, old_qwen, new_qwen)
        payload.update({
            "selected": {"lite_customer_weight": lite_weight, "qwen_customer_weight": qwen_weight},
            "lite": lite_metadata,
            "qwen": qwen_metadata,
            "routing_deployment": routing,
            "in_sample_smoke": smoke,
        })
        report_path = EVAL_DIR / "final_customer_adaptation_report.json"
        report_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
