from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from statistics import mean
from typing import Any

import joblib
import numpy as np
from scipy import sparse
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC

ROOT = Path(__file__).resolve().parents[1]
for import_path in (ROOT, ROOT / "backend"):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from scripts.lite_v2_feature_tune import Recipe as TextRecipe  # noqa: E402
from scripts.lite_v2_feature_tune import make_matrices  # noqa: E402
from scripts.lite_v2_sprint import aligned_probabilities, load_folds  # noqa: E402
from scripts.v5_dataset import load_v5_rows  # noqa: E402
from scripts.v5_experiment import (  # noqa: E402
    DEFAULT_CONTRACT,
    labels_for_view,
    load_json,
    operator_metrics,
)

MINILM_ID = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
SEED = 20260920


def load_qwen_teacher_embeddings(
    artifact: Path,
    rows: Sequence[Mapping[str, Any]],
) -> np.ndarray:
    """Load saved Qwen embeddings and prove they still match dataset row order."""
    bundle = joblib.load(artifact)
    pipeline = bundle["pipeline"]
    knn = pipeline.knn_model

    embeddings = np.asarray(knn._fit_X, dtype=np.float32)
    encoded_targets = np.asarray(knn._y)
    classes = np.asarray(knn.classes_, dtype=object)
    if np.issubdtype(encoded_targets.dtype, np.integer):
        stored_targets = classes[encoded_targets.astype(int)]
    else:
        stored_targets = encoded_targets.astype(object)

    expected_targets = np.asarray([str(row["category"]) for row in rows], dtype=object)
    if len(embeddings) != len(rows) or not np.array_equal(stored_targets, expected_targets):
        raise ValueError("Qwen teacher embedding alignment does not match the current dataset row order")
    return embeddings


def _normalize_rows(values: np.ndarray) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return np.divide(matrix, norms, out=np.zeros_like(matrix), where=norms > 0)


def fit_qwen_projection(
    student_train: np.ndarray,
    teacher_train: np.ndarray,
    student_validation: np.ndarray,
    *,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit a train-only linear student->Qwen map and return normalized projections."""
    projector = Ridge(alpha=float(alpha), fit_intercept=False)
    projector.fit(np.asarray(student_train, dtype=np.float32), np.asarray(teacher_train, dtype=np.float32))
    projected_train = _normalize_rows(projector.predict(student_train))
    projected_validation = _normalize_rows(projector.predict(student_validation))
    return projected_train, projected_validation


def _rows_fingerprint(rows: Sequence[Mapping[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(str(row.get("request_id") or "").encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(row.get("description") or "").encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def ensure_minilm_embeddings(
    rows: Sequence[Mapping[str, Any]],
    cache_path: Path,
    *,
    device: str,
    batch_size: int,
    max_seq_length: int,
) -> np.ndarray:
    manifest_path = cache_path.with_suffix(".json")
    fingerprint = _rows_fingerprint(rows)
    if cache_path.exists() and manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        cached = np.load(cache_path)
        if (
            manifest.get("model_id") == MINILM_ID
            and manifest.get("rows_fingerprint") == fingerprint
            and int(manifest.get("max_seq_length", 0)) == int(max_seq_length)
            and cached.shape[0] == len(rows)
        ):
            return _normalize_rows(cached)

    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(MINILM_ID, device=device, local_files_only=True)
    model.max_seq_length = int(max_seq_length)
    texts = [" ".join(str(row.get("description") or "").split()) for row in rows]
    started = time.perf_counter()
    embeddings = model.encode(
        texts,
        batch_size=int(batch_size),
        normalize_embeddings=True,
        show_progress_bar=True,
        convert_to_numpy=True,
    ).astype(np.float32, copy=False)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, embeddings)
    manifest_path.write_text(
        json.dumps(
            {
                "model_id": MINILM_ID,
                "rows": len(rows),
                "dimension": int(embeddings.shape[1]),
                "rows_fingerprint": fingerprint,
                "max_seq_length": int(max_seq_length),
                "device": device,
                "elapsed_seconds": round(time.perf_counter() - started, 3),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return embeddings


def _mean_metrics(records: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    keys = ("top1_accuracy", "top3_accuracy", "macro_f1", "true_label_mrr")
    return {key: round(mean(float(record[key]) for record in records), 9) for key in keys}


def _quality_key(metrics: Mapping[str, Any]) -> tuple[float, float, float, float]:
    return (
        float(metrics["top1_accuracy"]),
        float(metrics["macro_f1"]),
        float(metrics["top3_accuracy"]),
        float(metrics["true_label_mrr"]),
    )


def _prototype_probabilities(
    x_train: np.ndarray,
    y_train: Sequence[str],
    x_validation: np.ndarray,
    labels: Sequence[str],
    temperature: float = 0.08,
) -> np.ndarray:
    train = _normalize_rows(x_train)
    validation = _normalize_rows(x_validation)
    centroids = []
    y_array = np.asarray(y_train, dtype=object)
    for label in labels:
        mask = y_array == str(label)
        if not np.any(mask):
            centroids.append(np.zeros(train.shape[1], dtype=np.float32))
        else:
            centroids.append(_normalize_rows(train[mask].mean(axis=0, keepdims=True))[0])
    logits = validation @ np.asarray(centroids, dtype=np.float32).T
    logits = logits / max(float(temperature), 1e-6)
    logits -= logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    return exp / exp.sum(axis=1, keepdims=True)


def _knn_probabilities(
    x_train: np.ndarray,
    y_train: Sequence[str],
    x_validation: np.ndarray,
    labels: Sequence[str],
    neighbors: int = 11,
) -> np.ndarray:
    model = KNeighborsClassifier(
        n_neighbors=max(1, min(int(neighbors), len(y_train))),
        weights="distance",
        metric="cosine",
        algorithm="brute",
    ).fit(x_train, y_train)
    local = np.asarray(model.predict_proba(x_validation), dtype=float)
    result = np.zeros((len(x_validation), len(labels)), dtype=float)
    positions = {str(label): index for index, label in enumerate(labels)}
    for source, label in enumerate(model.classes_):
        target = positions.get(str(label))
        if target is not None:
            result[:, target] = local[:, source]
    sums = result.sum(axis=1, keepdims=True)
    return np.divide(result, sums, out=np.zeros_like(result), where=sums > 0)


def evaluate_frozen_minilm(
    *,
    output: Path,
    cache_path: Path,
    device: str,
    batch_size: int,
    max_seq_length: int,
) -> dict[str, Any]:
    rows = load_v5_rows()
    labels = labels_for_view(load_json(DEFAULT_CONTRACT), "full43")
    embeddings = ensure_minilm_embeddings(
        rows,
        cache_path,
        device=device,
        batch_size=batch_size,
        max_seq_length=max_seq_length,
    )
    row_index = {str(row["request_id"]): index for index, row in enumerate(rows)}
    folds = load_folds(rows)
    text_recipe = TextRecipe("word11", (1, 1), (3, 5))

    candidates: dict[str, list[dict[str, Any]]] = {"lite_v2": []}
    semantic_cs = (0.2, 0.5, 1.0, 2.0)
    blend_weights = (0.10, 0.20, 0.30, 0.40, 0.50)
    fusion_scales = (0.20, 0.50, 1.00, 2.00)
    fusion_cs = (0.20, 0.35, 0.50, 0.80)
    for c in semantic_cs:
        for family in ("svc", "svc_proto_knn"):
            for weight in blend_weights:
                candidates[f"blend_{family}_c{c}_w{weight}"] = []
    for scale in fusion_scales:
        for c in fusion_cs:
            candidates[f"fusion_scale{scale}_c{c}"] = []

    started_all = time.perf_counter()
    for fold_number, (repeat, fold, train_rows, validation_rows) in enumerate(folds, 1):
        print(f"fold {fold_number}/{len(folds)} repeat={repeat} fold={fold}", flush=True)
        y_train = [str(row["category"]) for row in train_rows]
        y_validation = [str(row["category"]) for row in validation_rows]
        train_indices = [row_index[str(row["request_id"])] for row in train_rows]
        validation_indices = [row_index[str(row["request_id"])] for row in validation_rows]
        e_train = embeddings[train_indices]
        e_validation = embeddings[validation_indices]

        x_train, x_validation = make_matrices(train_rows, validation_rows, text_recipe)
        base_model = LinearSVC(
            C=0.35,
            class_weight="balanced",
            max_iter=10000,
            random_state=SEED + repeat * 10 + fold,
        ).fit(x_train, y_train)
        base_probabilities = aligned_probabilities(base_model, x_validation, labels, "svc")
        candidates["lite_v2"].append(operator_metrics(y_validation, base_probabilities, labels))

        prototype = _prototype_probabilities(e_train, y_train, e_validation, labels)
        knn = _knn_probabilities(e_train, y_train, e_validation, labels)
        for c in semantic_cs:
            semantic_model = LinearSVC(
                C=c,
                class_weight="balanced",
                max_iter=10000,
                random_state=SEED + repeat * 10 + fold,
            ).fit(e_train, y_train)
            semantic = aligned_probabilities(semantic_model, e_validation, labels, "svc")
            semantic_ensemble = 0.90 * semantic + 0.05 * prototype + 0.05 * knn
            for family, semantic_probabilities in (
                ("svc", semantic),
                ("svc_proto_knn", semantic_ensemble),
            ):
                for weight in blend_weights:
                    blended = (1.0 - weight) * base_probabilities + weight * semantic_probabilities
                    candidates[f"blend_{family}_c{c}_w{weight}"].append(
                        operator_metrics(y_validation, blended, labels)
                    )

        for scale in fusion_scales:
            dense_train = sparse.csr_matrix(e_train * float(scale))
            dense_validation = sparse.csr_matrix(e_validation * float(scale))
            fused_train = sparse.hstack([x_train, dense_train], format="csr")
            fused_validation = sparse.hstack([x_validation, dense_validation], format="csr")
            for c in fusion_cs:
                model = LinearSVC(
                    C=c,
                    class_weight="balanced",
                    max_iter=10000,
                    random_state=SEED + repeat * 10 + fold,
                ).fit(fused_train, y_train)
                probabilities = aligned_probabilities(model, fused_validation, labels, "svc")
                candidates[f"fusion_scale{scale}_c{c}"].append(
                    operator_metrics(y_validation, probabilities, labels)
                )

    results = [
        {"candidate": name, "metrics": _mean_metrics(records)}
        for name, records in candidates.items()
    ]
    results.sort(key=lambda item: _quality_key(item["metrics"]), reverse=True)
    payload = {
        "stage": "frozen_minilm",
        "validation": "12 frozen grouped DEVELOPMENT folds",
        "lockbox_accessed": False,
        "rows": len(rows),
        "labels": len(labels),
        "minilm_model": MINILM_ID,
        "max_seq_length": int(max_seq_length),
        "baseline": next(item for item in results if item["candidate"] == "lite_v2"),
        "winner": results[0],
        "results": results,
        "elapsed_seconds": round(time.perf_counter() - started_all, 3),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"baseline": payload["baseline"], "winner": payload["winner"], "top10": results[:10]}, ensure_ascii=False, indent=2))
    return payload


def evaluate_qwen_distillation(
    *,
    output: Path,
    cache_path: Path,
    qwen_artifact: Path,
    device: str,
    batch_size: int,
    max_seq_length: int,
) -> dict[str, Any]:
    rows = load_v5_rows()
    labels = labels_for_view(load_json(DEFAULT_CONTRACT), "full43")
    student_embeddings = ensure_minilm_embeddings(
        rows,
        cache_path,
        device=device,
        batch_size=batch_size,
        max_seq_length=max_seq_length,
    )
    teacher_embeddings = _normalize_rows(load_qwen_teacher_embeddings(qwen_artifact, rows))
    row_index = {str(row["request_id"]): index for index, row in enumerate(rows)}
    folds = load_folds(rows)
    text_recipe = TextRecipe("word11", (1, 1), (3, 5))

    alphas = (0.1, 1.0, 10.0, 100.0)
    semantic_cs = (0.5, 1.0)
    blend_weights = (0.10, 0.20, 0.30, 0.40)
    fusion_scales = (0.20, 0.50, 1.00)
    fusion_cs = (0.20, 0.35, 0.50)
    candidates: dict[str, list[dict[str, Any]]] = {"lite_v2": []}
    projection_cosines: dict[str, list[float]] = {str(alpha): [] for alpha in alphas}
    for alpha in alphas:
        for c in semantic_cs:
            for weight in blend_weights:
                candidates[f"qdistill_a{alpha}_svc_c{c}_blend{weight}"] = []
        for scale in fusion_scales:
            for c in fusion_cs:
                candidates[f"qdistill_a{alpha}_fusion_s{scale}_c{c}"] = []

    started_all = time.perf_counter()
    for fold_number, (repeat, fold, train_rows, validation_rows) in enumerate(folds, 1):
        print(f"distill fold {fold_number}/{len(folds)} repeat={repeat} fold={fold}", flush=True)
        y_train = [str(row["category"]) for row in train_rows]
        y_validation = [str(row["category"]) for row in validation_rows]
        train_indices = [row_index[str(row["request_id"])] for row in train_rows]
        validation_indices = [row_index[str(row["request_id"])] for row in validation_rows]
        student_train = student_embeddings[train_indices]
        student_validation = student_embeddings[validation_indices]
        teacher_train = teacher_embeddings[train_indices]
        teacher_validation = teacher_embeddings[validation_indices]

        x_train, x_validation = make_matrices(train_rows, validation_rows, text_recipe)
        base_model = LinearSVC(
            C=0.35,
            class_weight="balanced",
            max_iter=10000,
            random_state=SEED + repeat * 10 + fold,
        ).fit(x_train, y_train)
        base_probabilities = aligned_probabilities(base_model, x_validation, labels, "svc")
        candidates["lite_v2"].append(operator_metrics(y_validation, base_probabilities, labels))

        for alpha in alphas:
            projected_train, projected_validation = fit_qwen_projection(
                student_train,
                teacher_train,
                student_validation,
                alpha=alpha,
            )
            cosine = np.sum(projected_validation * teacher_validation, axis=1)
            projection_cosines[str(alpha)].append(float(np.mean(cosine)))

            for c in semantic_cs:
                semantic_model = LinearSVC(
                    C=c,
                    class_weight="balanced",
                    max_iter=10000,
                    random_state=SEED + repeat * 10 + fold,
                ).fit(projected_train, y_train)
                semantic_probabilities = aligned_probabilities(
                    semantic_model,
                    projected_validation,
                    labels,
                    "svc",
                )
                for weight in blend_weights:
                    probabilities = (
                        (1.0 - weight) * base_probabilities + weight * semantic_probabilities
                    )
                    candidates[f"qdistill_a{alpha}_svc_c{c}_blend{weight}"].append(
                        operator_metrics(y_validation, probabilities, labels)
                    )

            for scale in fusion_scales:
                projected_train_sparse = sparse.csr_matrix(projected_train * float(scale))
                projected_validation_sparse = sparse.csr_matrix(
                    projected_validation * float(scale)
                )
                fused_train = sparse.hstack([x_train, projected_train_sparse], format="csr")
                fused_validation = sparse.hstack(
                    [x_validation, projected_validation_sparse],
                    format="csr",
                )
                for c in fusion_cs:
                    model = LinearSVC(
                        C=c,
                        class_weight="balanced",
                        max_iter=10000,
                        random_state=SEED + repeat * 10 + fold,
                    ).fit(fused_train, y_train)
                    probabilities = aligned_probabilities(
                        model,
                        fused_validation,
                        labels,
                        "svc",
                    )
                    candidates[f"qdistill_a{alpha}_fusion_s{scale}_c{c}"].append(
                        operator_metrics(y_validation, probabilities, labels)
                    )

    results = [
        {"candidate": name, "metrics": _mean_metrics(records)}
        for name, records in candidates.items()
    ]
    results.sort(key=lambda item: _quality_key(item["metrics"]), reverse=True)
    payload = {
        "stage": "qwen_embedding_distillation",
        "validation": "12 frozen grouped DEVELOPMENT folds; projection fit on outer TRAIN only",
        "lockbox_accessed": False,
        "rows": len(rows),
        "labels": len(labels),
        "student_model": MINILM_ID,
        "teacher": "saved Qwen3-Embedding-4B 2560-d embeddings",
        "projection_mean_cosine": {
            alpha: round(mean(values), 6) for alpha, values in projection_cosines.items()
        },
        "baseline": next(item for item in results if item["candidate"] == "lite_v2"),
        "winner": results[0],
        "results": results,
        "elapsed_seconds": round(time.perf_counter() - started_all, 3),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "baseline": payload["baseline"],
                "winner": payload["winner"],
                "projection_mean_cosine": payload["projection_mean_cosine"],
                "top10": results[:10],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Local Lite V3 quality sprint using cached semantic teachers")
    parser.add_argument(
        "--stage",
        choices=("frozen-minilm", "qwen-distill"),
        default="frozen-minilm",
    )
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/lite_v3/frozen_minilm.json")
    parser.add_argument("--cache", type=Path, default=ROOT / "outputs/lite_v3/minilm_256.npy")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-seq-length", type=int, default=256)
    parser.add_argument(
        "--qwen-artifact",
        type=Path,
        default=ROOT / "models/v5/category_qwen4b_lite.joblib",
    )
    args = parser.parse_args()
    if args.stage == "frozen-minilm":
        evaluate_frozen_minilm(
            output=args.output,
            cache_path=args.cache,
            device=args.device,
            batch_size=args.batch_size,
            max_seq_length=args.max_seq_length,
        )
    elif args.stage == "qwen-distill":
        evaluate_qwen_distillation(
            output=args.output,
            cache_path=args.cache,
            qwen_artifact=args.qwen_artifact,
            device=args.device,
            batch_size=args.batch_size,
            max_seq_length=args.max_seq_length,
        )


if __name__ == "__main__":
    main()
