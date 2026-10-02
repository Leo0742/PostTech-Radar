from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

from app.core.config import DATABASE_PATH  # noqa: E402
from app.ml.v4.evaluation import REGISTRATION_FIELDS, classification_metrics, safe_registration_row  # noqa: E402
from app.services.data_service import load_rows  # noqa: E402

META_FIELDS = tuple(field for field in REGISTRATION_FIELDS if field != "description")


def _instruction(model_id: str) -> str | None:
    name = model_id.lower()
    if "qwen3-embedding" in name:
        return "Classify a Russian Service Desk ticket by its operational issue category."
    if "e5-large-instruct" in name:
        return "Given a Russian Service Desk ticket, retrieve and classify tickets with the same issue category."
    return None


def _fold_matrix(
    embeddings: np.ndarray,
    metadata: np.ndarray,
    train: np.ndarray,
    validation: np.ndarray,
    metadata_scale: float,
) -> tuple[sparse.csr_matrix, sparse.csr_matrix]:
    if metadata_scale == 0:
        return sparse.csr_matrix(embeddings[train]), sparse.csr_matrix(embeddings[validation])
    encoder = OneHotEncoder(handle_unknown="ignore", min_frequency=2)
    train_meta = encoder.fit_transform(metadata[train]) * metadata_scale
    validation_meta = encoder.transform(metadata[validation]) * metadata_scale
    return (
        sparse.hstack([sparse.csr_matrix(embeddings[train]), train_meta], format="csr"),
        sparse.hstack([sparse.csr_matrix(embeddings[validation]), validation_meta], format="csr"),
    )


def _score(metrics: dict[str, Any], view: str) -> float:
    if view == "top15":
        return metrics["macro_f1"] + 0.25 * metrics["accuracy"] + 0.15 * metrics["worst_class_f1"]
    bands = metrics["support_bands"]
    rare = bands["1-5"]["macro_f1"] or 0.0
    low = bands["5-20"]["macro_f1"] or 0.0
    return metrics["macro_f1"] + 0.25 * metrics["balanced_accuracy"] + 0.2 * rare + 0.1 * low


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--slug", required=True)
    parser.add_argument("--view", choices=("top15", "full43"), required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, default=Path(os.environ.get("HF_HOME", "/workspace/hf-cache")))
    args = parser.parse_args()
    import torch
    from huggingface_hub import model_info
    from sentence_transformers import SentenceTransformer

    pointer = json.loads((PROJECT_ROOT / "artifacts/gpu_research_v4/protocol.json").read_text())
    protocol = json.loads((PROJECT_ROOT / pointer["protocol_path"]).read_text())
    all_rows = [row for row in load_rows(DATABASE_PATH) if str(row.get("description") or "").strip()]
    counts = Counter(str(row["category"]) for row in all_rows)
    view_labels = {label for label, _ in counts.most_common(15)} if args.view == "top15" else set(counts)
    rows = [row for row in all_rows if str(row["category"]) in view_labels]
    by_id = {str(row["request_id"]): index for index, row in enumerate(rows)}
    labels = np.asarray([str(row["category"]) for row in rows], dtype=object)
    metadata = np.asarray(
        [[safe_registration_row(row)[field] for field in META_FIELDS] for row in rows], dtype=object
    )
    info = model_info(args.model_id)
    revision = str(info.sha)
    torch.cuda.reset_peak_memory_stats()
    loaded = time.perf_counter()
    model = SentenceTransformer(
        args.model_id,
        revision=revision,
        cache_folder=str(args.cache_dir),
        trust_remote_code=False,
        model_kwargs={"torch_dtype": torch.bfloat16},
    )
    load_seconds = time.perf_counter() - loaded
    kwargs: dict[str, Any] = {
        "batch_size": args.batch_size,
        "show_progress_bar": True,
        "convert_to_numpy": True,
        "normalize_embeddings": True,
    }
    instruction = _instruction(args.model_id)
    if instruction:
        kwargs["prompt"] = f"Instruct: {instruction}\nQuery:"
    encoded = time.perf_counter()
    embeddings = np.asarray(model.encode([str(row["description"]) for row in rows], **kwargs), dtype=np.float32)
    encode_seconds = time.perf_counter() - encoded
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output.with_suffix(".embeddings.npz"),
        request_ids=np.asarray([str(row["request_id"]) for row in rows], dtype=str),
        embeddings=embeddings,
    )

    grid = [
        {"C": c, "metadata_scale": scale, "class_weight": weight}
        for c in (0.3, 1.0, 3.0)
        for scale in (0.0, 0.25, 0.5, 1.0)
        for weight in (None, "balanced")
    ]
    tuning = []
    for config in grid:
        truth: list[str] = []
        predicted: list[str] = []
        for fold in protocol["folds"]:
            # Use one frozen development fold for the broad hyperparameter grid,
            # then confirm the selected recipe on every fold of all three repeats.
            if int(fold["repeat"]) != 0 or int(fold["fold"]) != 0:
                continue
            train = np.asarray([by_id[item] for item in fold["train_request_ids"] if item in by_id], dtype=int)
            validation = np.asarray([by_id[item] for item in fold["validation_request_ids"] if item in by_id], dtype=int)
            x_train, x_validation = _fold_matrix(
                embeddings, metadata, train, validation, float(config["metadata_scale"])
            )
            estimator = LogisticRegression(
                C=float(config["C"]),
                max_iter=3000,
                class_weight=config["class_weight"],
                random_state=20260917,
            )
            estimator.fit(x_train, labels[train])
            truth.extend(str(labels[index]) for index in validation)
            predicted.extend(str(value) for value in estimator.predict(x_validation))
        metrics = classification_metrics(truth, predicted, labels=sorted(view_labels))
        tuning.append({"parameters": config, "objective": round(_score(metrics, args.view), 8), "metrics": metrics})
    selected = max(tuning, key=lambda row: row["objective"])

    confirmation: dict[int, dict[str, Any]] = {}
    all_predictions = []
    for repeat in (0, 1, 2):
        truth = []
        predicted = []
        request_ids = []
        for fold in protocol["folds"]:
            if int(fold["repeat"]) != repeat:
                continue
            train = np.asarray([by_id[item] for item in fold["train_request_ids"] if item in by_id], dtype=int)
            validation = np.asarray([by_id[item] for item in fold["validation_request_ids"] if item in by_id], dtype=int)
            x_train, x_validation = _fold_matrix(
                embeddings, metadata, train, validation, float(selected["parameters"]["metadata_scale"])
            )
            estimator = LogisticRegression(
                C=float(selected["parameters"]["C"]),
                max_iter=3000,
                class_weight=selected["parameters"]["class_weight"],
                random_state=20260917 + repeat,
            )
            estimator.fit(x_train, labels[train])
            values = [str(value) for value in estimator.predict(x_validation)]
            truth.extend(str(labels[index]) for index in validation)
            predicted.extend(values)
            request_ids.extend(str(rows[index]["request_id"]) for index in validation)
        confirmation[repeat] = classification_metrics(truth, predicted, labels=sorted(view_labels))
        all_predictions.extend(
            {"repeat": repeat, "request_id": item, "truth": target, "prediction": value}
            for item, target, value in zip(request_ids, truth, predicted, strict=True)
        )
    summary = defaultdict(list)
    for metrics in confirmation.values():
        for key in ("accuracy", "macro_f1", "weighted_f1", "balanced_accuracy", "worst_class_f1"):
            summary[key].append(metrics[key])
    stability = {
        key: {"mean": round(float(np.mean(values)), 6), "std": round(float(np.std(values)), 6)}
        for key, values in summary.items()
    }
    temporal_train = np.asarray(
        [by_id[item] for item in protocol["temporal_train_request_ids"] if item in by_id], dtype=int
    )
    temporal_test = np.asarray(
        [by_id[item] for item in protocol["temporal_test_request_ids"] if item in by_id], dtype=int
    )
    temporal_known = {str(labels[index]) for index in temporal_train}
    temporal_test = np.asarray(
        [index for index in temporal_test if str(labels[index]) in temporal_known], dtype=int
    )
    temporal_x_train, temporal_x_test = _fold_matrix(
        embeddings,
        metadata,
        temporal_train,
        temporal_test,
        float(selected["parameters"]["metadata_scale"]),
    )
    temporal_model = LogisticRegression(
        C=float(selected["parameters"]["C"]),
        max_iter=3000,
        class_weight=selected["parameters"]["class_weight"],
        random_state=20260920,
    )
    temporal_model.fit(temporal_x_train, labels[temporal_train])
    temporal_predictions = [str(value) for value in temporal_model.predict(temporal_x_test)]
    temporal = {
        "train_rows": len(temporal_train),
        "test_rows": len(temporal_test),
        "metrics": classification_metrics(
            [str(labels[index]) for index in temporal_test], temporal_predictions, labels=sorted(view_labels)
        ),
    }
    payload = {
        "candidate_id": f"category/{args.slug}-deep/{args.view}/v4",
        "status": "MEASURED_DEEP_OPTIMIZATION",
        "model_id": args.model_id,
        "revision": revision,
        "license": str((info.card_data or {}).get("license") or "UNKNOWN"),
        "trust_remote_code": False,
        "view": args.view,
        "dataset_sha256": protocol["dataset_sha256"],
        "split_sha256": protocol["split_sha256"],
        "sealed_holdout_accessed": False,
        "real_only_evaluation": True,
        "rows": len(rows),
        "grid_size": len(grid),
        "selected": selected,
        "repeat_metrics": confirmation,
        "stability": stability,
        "temporal": temporal,
        "load_seconds": round(load_seconds, 3),
        "encode_seconds": round(encode_seconds, 3),
        "encode_ms_per_ticket": round(1000 * encode_seconds / len(rows), 4),
        "peak_vram_bytes": int(torch.cuda.max_memory_allocated()),
        "tuning": tuning,
        "predictions": all_predictions,
    }
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.view == "full43":
        metadata_scale = float(selected["parameters"]["metadata_scale"])
        deployment_encoder = OneHotEncoder(handle_unknown="ignore", min_frequency=2)
        deployment_metadata = deployment_encoder.fit_transform(metadata) * metadata_scale
        deployment_matrix = sparse.hstack(
            [sparse.csr_matrix(embeddings), deployment_metadata], format="csr"
        ) if metadata_scale else sparse.csr_matrix(embeddings)
        deployment_classifier = LogisticRegression(
            C=float(selected["parameters"]["C"]),
            max_iter=3000,
            class_weight=selected["parameters"]["class_weight"],
            random_state=20260920,
        )
        deployment_classifier.fit(deployment_matrix, labels)
        joblib.dump(
            {
                "model_id": args.model_id,
                "revision": revision,
                "license": str((info.card_data or {}).get("license") or "UNKNOWN"),
                "metadata_fields": META_FIELDS,
                "metadata_scale": metadata_scale,
                "metadata_encoder": deployment_encoder,
                "classifier": deployment_classifier,
                "instruction": instruction,
                "selected_parameters": selected["parameters"],
                "dataset_sha256": protocol["dataset_sha256"],
                "split_sha256": protocol["split_sha256"],
            },
            args.output.with_suffix(".joblib"),
        )
    print(json.dumps({"output": str(args.output), "selected": selected["parameters"], "stability": stability}, ensure_ascii=False))


if __name__ == "__main__":
    main()
