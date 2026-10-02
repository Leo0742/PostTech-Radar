from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import OneHotEncoder

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

from app.core.config import DATABASE_PATH  # noqa: E402
from app.ml.v4.evaluation import REGISTRATION_FIELDS, classification_metrics, safe_registration_row  # noqa: E402
from app.services.data_service import load_rows  # noqa: E402

METADATA_FIELDS = tuple(field for field in REGISTRATION_FIELDS if field != "description")


def _load_protocol() -> dict[str, Any]:
    pointer = json.loads((PROJECT_ROOT / "artifacts/gpu_research_v4/protocol.json").read_text(encoding="utf-8"))
    return json.loads((PROJECT_ROOT / pointer["protocol_path"]).read_text(encoding="utf-8"))


def _instruction(model_id: str) -> str | None:
    lowered = model_id.lower()
    if "qwen3-embedding" in lowered:
        return "Classify a Russian Service Desk ticket by its operational issue category."
    if "e5-large-instruct" in lowered:
        return "Given a Russian Service Desk ticket, retrieve and classify tickets with the same issue category."
    return None


def _encode(model: Any, texts: list[str], model_id: str, batch_size: int) -> np.ndarray:
    instruction = _instruction(model_id)
    kwargs: dict[str, Any] = {
        "batch_size": batch_size,
        "show_progress_bar": True,
        "convert_to_numpy": True,
        "normalize_embeddings": True,
    }
    if instruction:
        kwargs["prompt"] = f"Instruct: {instruction}\nQuery:"
    return np.asarray(model.encode(texts, **kwargs), dtype=np.float32)


def _metadata(rows: list[dict[str, Any]]) -> np.ndarray:
    return np.asarray([[safe_registration_row(row)[field] for field in METADATA_FIELDS] for row in rows], dtype=object)


def _mapped_predictions(estimator: Any, x_test: Any) -> list[str]:
    return [str(value) for value in estimator.predict(x_test)]


def _run_fold_models(
    embeddings: np.ndarray,
    metadata: np.ndarray,
    labels: np.ndarray,
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
    seed: int,
) -> dict[str, list[str]]:
    y_train = labels[train_indices]
    text_lr = LogisticRegression(max_iter=2500, class_weight="balanced", random_state=seed)
    text_lr.fit(embeddings[train_indices], y_train)

    encoder = OneHotEncoder(handle_unknown="ignore", min_frequency=2)
    train_metadata = encoder.fit_transform(metadata[train_indices])
    validation_metadata = encoder.transform(metadata[validation_indices])
    combined_train = sparse.hstack([sparse.csr_matrix(embeddings[train_indices]), train_metadata], format="csr")
    combined_validation = sparse.hstack(
        [sparse.csr_matrix(embeddings[validation_indices]), validation_metadata], format="csr"
    )
    combined_lr = LogisticRegression(max_iter=2500, class_weight="balanced", random_state=seed)
    combined_lr.fit(combined_train, y_train)

    neighbors = min(7, len(train_indices))
    knn = KNeighborsClassifier(n_neighbors=neighbors, weights="distance", metric="cosine", algorithm="brute")
    knn.fit(embeddings[train_indices], y_train)

    centroids = {
        label: embeddings[train_indices][y_train == label].mean(axis=0)
        for label in sorted(set(y_train))
    }
    centroid_labels = np.asarray(list(centroids))
    centroid_matrix = np.asarray([centroids[label] for label in centroid_labels])
    centroid_matrix /= np.maximum(np.linalg.norm(centroid_matrix, axis=1, keepdims=True), 1e-12)
    centroid_scores = embeddings[validation_indices] @ centroid_matrix.T
    return {
        "embedding_lr": _mapped_predictions(text_lr, embeddings[validation_indices]),
        "embedding_metadata_lr": _mapped_predictions(combined_lr, combined_validation),
        "embedding_knn": _mapped_predictions(knn, embeddings[validation_indices]),
        "embedding_centroid": [str(value) for value in centroid_labels[centroid_scores.argmax(axis=1)]],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark a frozen GPU embedding model on v4 OOF folds")
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--revision", default="main")
    parser.add_argument("--slug", required=True)
    parser.add_argument("--view", choices=("top15", "full43"), required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--repeat", type=int, default=0)
    parser.add_argument("--cache-dir", type=Path, default=Path(os.environ.get("HF_HOME", "/workspace/hf-cache")))
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "artifacts/gpu_research_v4/embeddings")
    parser.add_argument("--export-winner", action="store_true")
    args = parser.parse_args()

    import torch
    from huggingface_hub import model_info
    from sentence_transformers import SentenceTransformer

    protocol = _load_protocol()
    all_rows = [row for row in load_rows(DATABASE_PATH) if str(row.get("description") or "").strip()]
    counts = Counter(str(row["category"]) for row in all_rows)
    view_labels = (
        {label for label, _count in counts.most_common(15)} if args.view == "top15" else set(counts)
    )
    development_ids = set(protocol["development_request_ids"])
    rows = [row for row in all_rows if str(row["request_id"]) in development_ids and str(row["category"]) in view_labels]
    by_id = {str(row["request_id"]): index for index, row in enumerate(rows)}
    labels = np.asarray([str(row["category"]) for row in rows], dtype=object)
    texts = [str(row["description"]) for row in rows]
    metadata = _metadata(rows)

    info = model_info(args.model_id, revision=args.revision)
    resolved_revision = str(info.sha)
    license_name = str((info.card_data or {}).get("license") or "UNKNOWN")
    torch.cuda.reset_peak_memory_stats()
    load_started = time.perf_counter()
    model = SentenceTransformer(
        args.model_id,
        revision=resolved_revision,
        cache_folder=str(args.cache_dir),
        trust_remote_code=False,
        model_kwargs={"torch_dtype": torch.bfloat16},
    )
    load_seconds = time.perf_counter() - load_started
    encode_started = time.perf_counter()
    embeddings = _encode(model, texts, args.model_id, args.batch_size)
    encode_seconds = time.perf_counter() - encode_started

    truth_by_model: dict[str, list[str]] = {}
    predicted_by_model: dict[str, list[str]] = {}
    ids_by_model: dict[str, list[str]] = {}
    fold_records = []
    for fold in protocol["folds"]:
        if int(fold["repeat"]) != args.repeat:
            continue
        train_indices = np.asarray([by_id[item] for item in fold["train_request_ids"] if item in by_id], dtype=int)
        validation_indices = np.asarray(
            [by_id[item] for item in fold["validation_request_ids"] if item in by_id], dtype=int
        )
        if not len(train_indices) or not len(validation_indices):
            continue
        predictions = _run_fold_models(
            embeddings,
            metadata,
            labels,
            train_indices,
            validation_indices,
            int(fold["seed"]),
        )
        fold_records.append(
            {
                "fold": int(fold["fold"]),
                "train": len(train_indices),
                "validation": len(validation_indices),
            }
        )
        for name, values in predictions.items():
            truth_by_model.setdefault(name, []).extend(str(labels[index]) for index in validation_indices)
            predicted_by_model.setdefault(name, []).extend(values)
            ids_by_model.setdefault(name, []).extend(str(rows[index]["request_id"]) for index in validation_indices)

    ordered_labels = sorted(view_labels)
    metrics = {
        name: classification_metrics(truth_by_model[name], values, labels=ordered_labels)
        for name, values in predicted_by_model.items()
    }
    payload = {
        "candidate_id": f"category/{args.slug}-frozen/{args.view}/v4",
        "status": "MEASURED_GPU_OOF",
        "model_id": args.model_id,
        "revision": resolved_revision,
        "license": license_name,
        "trust_remote_code": False,
        "view": args.view,
        "dataset_sha256": protocol["dataset_sha256"],
        "split_sha256": protocol["split_sha256"],
        "sealed_holdout_accessed": False,
        "real_only_evaluation": True,
        "device": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "embedding_dimension": int(embeddings.shape[1]),
        "rows": len(rows),
        "load_seconds": round(load_seconds, 3),
        "encode_seconds": round(encode_seconds, 3),
        "encode_ms_per_ticket": round(1000 * encode_seconds / len(rows), 4),
        "peak_vram_bytes": int(torch.cuda.max_memory_allocated()),
        "folds": fold_records,
        "metrics": metrics,
        "predictions": {
            name: [
                {"request_id": request_id, "truth": truth, "prediction": prediction}
                for request_id, truth, prediction in zip(
                    ids_by_model[name], truth_by_model[name], predicted_by_model[name], strict=True
                )
            ]
            for name in metrics
        },
    }
    output = args.output_root / f"{args.slug}-{args.view}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.export_winner:
        artifact = args.output_root / f"{args.slug}-{args.view}-embeddings.joblib"
        joblib.dump({"request_ids": [str(row["request_id"]) for row in rows], "embeddings": embeddings}, artifact)
    print(json.dumps({"output": str(output), "revision": resolved_revision, "metrics": metrics}, ensure_ascii=False))


if __name__ == "__main__":
    main()
