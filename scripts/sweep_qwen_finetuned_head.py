from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import LinearSVC

ROOT = Path(__file__).resolve().parents[1]
for import_path in (ROOT, ROOT / "backend"):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from app.ml.v5_runtime import build_v5_structured_features  # noqa: E402
from scripts.finalize_customer_adaptation import (  # noqa: E402
    aligned,
    load_all,
    repeated_knn_training,
    softmax,
    topk_metrics,
    weighted_centroids,
)
from scripts.gpu_customer_finetune import OUT_DIR  # noqa: E402


CONFIG = "qwen_r16_mnrl_s80"
SEED = 20260921


def main() -> None:
    old_rows, new_rows, *_rest = load_all()
    all_rows = old_rows + new_rows
    by_id = {str(row["request_id"]): row for row in all_rows}
    new_ids = {str(row["request_id"]) for row in new_rows}
    labels = sorted({str(row["category"]) for row in all_rows})
    baseline = json.loads((OUT_DIR / "qwen_base_finetune_blend.json").read_text(encoding="utf-8"))["baseline"]

    c_values = [0.5, 1.0, 2.0]
    metadata_scales = [0.35, 0.75, 1.0]
    customer_weights = [8.0, 16.0, 24.0]
    svc_temperatures = [0.7, 1.0, 1.3]
    prototype_temperatures = [0.06, 0.08, 0.12]
    knn_neighbors = [7, 11, 15]
    blends = [
        (1.0, 0.0, 0.0),
        (0.95, 0.025, 0.025),
        (0.90, 0.05, 0.05),
        (0.85, 0.10, 0.05),
        (0.85, 0.05, 0.10),
        (0.80, 0.10, 0.10),
    ]

    folds = []
    for fold in (0, 1):
        data = np.load(OUT_DIR / f"{CONFIG}_fold{fold}_embeddings.npz")
        train_ids = [str(value) for value in data["train_ids"].tolist()]
        valid_ids = [str(value) for value in data["valid_ids"].tolist()]
        folds.append({
            "train_rows": [by_id[value] for value in train_ids],
            "valid_rows": [by_id[value] for value in valid_ids],
            "train_embeddings": np.asarray(data["train_embeddings"], dtype=np.float32),
            "valid_embeddings": np.asarray(data["valid_embeddings"], dtype=np.float32),
            "truth": [str(by_id[value]["category"]) for value in valid_ids],
            "train_ids": train_ids,
        })

    component_cache: dict[tuple, list[dict[str, np.ndarray]]] = {}
    for customer_weight in customer_weights:
        for c_value in c_values:
            for metadata_scale in metadata_scales:
                fold_components = []
                for fold in folds:
                    train_rows = fold["train_rows"]
                    valid_rows = fold["valid_rows"]
                    train_embeddings = fold["train_embeddings"]
                    valid_embeddings = fold["valid_embeddings"]
                    targets = [str(row["category"]) for row in train_rows]
                    weights = np.asarray([
                        customer_weight if request_id in new_ids else 1.0
                        for request_id in fold["train_ids"]
                    ], dtype=float)

                    encoder = OneHotEncoder(handle_unknown="ignore", min_frequency=2, sparse_output=False)
                    train_meta = np.asarray(encoder.fit_transform(build_v5_structured_features(train_rows)), dtype=float)
                    valid_meta = np.asarray(encoder.transform(build_v5_structured_features(valid_rows)), dtype=float)
                    train_x = np.concatenate([train_embeddings, train_meta * metadata_scale], axis=1)
                    valid_x = np.concatenate([valid_embeddings, valid_meta * metadata_scale], axis=1)
                    classifier = LinearSVC(
                        C=c_value,
                        class_weight="balanced",
                        random_state=SEED,
                        max_iter=10000,
                    )
                    classifier.fit(train_x, targets, sample_weight=weights)
                    raw_scores = classifier.decision_function(valid_x)

                    centroids = weighted_centroids(train_embeddings, targets, weights, labels)
                    prototype_scores = valid_embeddings @ centroids.T
                    knn_x, knn_y = repeated_knn_training(train_embeddings, targets, weights)
                    knn_by_k = {}
                    for k in knn_neighbors:
                        model = KNeighborsClassifier(
                            n_neighbors=max(1, min(k, len(knn_y))),
                            weights="distance",
                            metric="cosine",
                            algorithm="brute",
                        ).fit(knn_x, knn_y)
                        knn_by_k[k] = aligned(model.predict_proba(valid_embeddings), model.classes_, labels)

                    fold_components.append({
                        "raw_scores": raw_scores,
                        "classes": np.asarray(classifier.classes_, dtype=object),
                        "prototype_scores": prototype_scores,
                        "knn": knn_by_k,
                    })
                component_cache[(customer_weight, c_value, metadata_scale)] = fold_components

    results = []
    for customer_weight in customer_weights:
        for c_value in c_values:
            for metadata_scale in metadata_scales:
                components = component_cache[(customer_weight, c_value, metadata_scale)]
                for svc_temperature in svc_temperatures:
                    for prototype_temperature in prototype_temperatures:
                        for k in knn_neighbors:
                            for supervised_weight, prototype_weight, knn_weight in blends:
                                truths: list[str] = []
                                probabilities = []
                                for fold, comp in zip(folds, components, strict=True):
                                    supervised = aligned(
                                        softmax(comp["raw_scores"], svc_temperature),
                                        comp["classes"],
                                        labels,
                                    )
                                    prototype = softmax(comp["prototype_scores"], prototype_temperature)
                                    probs = (
                                        supervised_weight * supervised
                                        + prototype_weight * prototype
                                        + knn_weight * comp["knn"][k]
                                    )
                                    probs /= probs.sum(axis=1, keepdims=True)
                                    truths.extend(fold["truth"])
                                    probabilities.append(probs)
                                metrics = topk_metrics(truths, np.vstack(probabilities), labels)
                                results.append({
                                    "c": c_value,
                                    "metadata_scale": metadata_scale,
                                    "customer_weight": customer_weight,
                                    "svc_temperature": svc_temperature,
                                    "prototype_temperature": prototype_temperature,
                                    "knn_neighbors": k,
                                    "blend": {
                                        "supervised": supervised_weight,
                                        "prototype": prototype_weight,
                                        "knn": knn_weight,
                                    },
                                    **metrics,
                                })

    results.sort(
        key=lambda row: (row["top1"], row["top3"], row["macro_f1_present_truth"]),
        reverse=True,
    )
    guarded = [
        row for row in results
        if row["top1"] >= baseline["top1"]
        and row["top3"] >= baseline["top3"]
        and row["macro_f1_present_truth"] >= baseline["macro_f1_present_truth"]
    ]
    payload = {
        "method": "same saved fine-tuned embeddings from 2-fold customer OOF; head-only sweep; no lockbox",
        "baseline": baseline,
        "winner": results[0],
        "guarded_winner": guarded[0] if guarded else None,
        "guarded_count": len(guarded),
        "top20": results[:20],
        "lockbox_accessed": False,
    }
    (OUT_DIR / "qwen_finetuned_head_sweep.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
