from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
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
from scripts.gpu_customer_finetune import (  # noqa: E402
    Config,
    OUT_DIR,
    _encode,
    _free_model,
    _train_encoder,
)


FOLDS_PATH = ROOT / "artifacts/gpu_research_v5/protocol/folds.json"
PROTOCOL_PATH = ROOT / "artifacts/gpu_research_v5/protocol/protocol.json"
RUN_DIR = OUT_DIR / "historical_guarded_r16_mnrl_s80"
CONFIG = Config("qwen_r16_mnrl_s80", 16, 2e-5, 80, "mnrl", 256, 8)

HEAD = {
    "c": 1.0,
    "metadata_scale": 0.75,
    "customer_weight": 24.0,
    "svc_temperature": 1.0,
    "prototype_temperature": 0.08,
    "knn_neighbors": 7,
    "blend": {"supervised": 0.95, "prototype": 0.025, "knn": 0.025},
}


def predict_head(
    train_rows: list[dict],
    train_embeddings: np.ndarray,
    valid_rows: list[dict],
    valid_embeddings: np.ndarray,
    labels: list[str],
    customer_ids: set[str],
    seed: int,
) -> np.ndarray:
    targets = [str(row["category"]) for row in train_rows]
    weights = np.asarray([
        HEAD["customer_weight"] if str(row["request_id"]) in customer_ids else 1.0
        for row in train_rows
    ], dtype=float)

    encoder = OneHotEncoder(handle_unknown="ignore", min_frequency=2, sparse_output=False)
    train_meta = np.asarray(encoder.fit_transform(build_v5_structured_features(train_rows)), dtype=float)
    valid_meta = np.asarray(encoder.transform(build_v5_structured_features(valid_rows)), dtype=float)
    train_x = np.concatenate([train_embeddings, train_meta * HEAD["metadata_scale"]], axis=1)
    valid_x = np.concatenate([valid_embeddings, valid_meta * HEAD["metadata_scale"]], axis=1)

    classifier = LinearSVC(
        C=HEAD["c"],
        class_weight="balanced",
        random_state=seed,
        max_iter=10000,
    )
    classifier.fit(train_x, targets, sample_weight=weights)
    supervised = aligned(
        softmax(classifier.decision_function(valid_x), HEAD["svc_temperature"]),
        classifier.classes_,
        labels,
    )

    centroids = weighted_centroids(train_embeddings, targets, weights, labels)
    prototype = softmax(valid_embeddings @ centroids.T, HEAD["prototype_temperature"])

    knn_x, knn_y = repeated_knn_training(train_embeddings, targets, weights)
    knn = KNeighborsClassifier(
        n_neighbors=max(1, min(int(HEAD["knn_neighbors"]), len(knn_y))),
        weights="distance",
        metric="cosine",
        algorithm="brute",
    ).fit(knn_x, knn_y)
    knn_probabilities = aligned(knn.predict_proba(valid_embeddings), knn.classes_, labels)

    blend = HEAD["blend"]
    result = (
        blend["supervised"] * supervised
        + blend["prototype"] * prototype
        + blend["knn"] * knn_probabilities
    )
    return result / result.sum(axis=1, keepdims=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    old_rows, new_rows, *_ = load_all()
    old_by_id = {str(row["request_id"]): row for row in old_rows}
    new_ids = {str(row["request_id"]) for row in new_rows}
    labels = sorted({str(row["category"]) for row in old_rows + new_rows})
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    lockbox_ids = {str(value) for value in protocol["internal_lockbox_request_ids"]}
    folds = json.loads(FOLDS_PATH.read_text(encoding="utf-8"))

    if new_ids & set(old_by_id):
        raise RuntimeError("Customer IDs overlap historical IDs")
    RUN_DIR.mkdir(parents=True, exist_ok=True)

    for index, fold in enumerate(folds):
        train_ids = [str(value) for value in fold["train_request_ids"]]
        valid_ids = [str(value) for value in fold["validation_request_ids"]]
        if set(train_ids) & lockbox_ids or set(valid_ids) & lockbox_ids:
            raise RuntimeError(f"Fold {index} touches the internal lockbox")
        output = RUN_DIR / f"fold_{index:02d}.json"
        if output.exists() and not args.force:
            print(f"SKIP fold={index} existing={output.name}", flush=True)
            continue

        train_old = [old_by_id[value] for value in train_ids]
        valid_rows = [old_by_id[value] for value in valid_ids]
        train_rows = train_old + new_rows
        seed = int(fold.get("seed", 20260921)) + 7000
        print(
            f"START historical fold={index + 1}/{len(folds)} repeat={fold['repeat']} fold={fold['fold']} "
            f"train={len(train_rows)} valid={len(valid_rows)}",
            flush=True,
        )
        model, train_stats = _train_encoder("qwen", CONFIG, train_rows, new_ids, seed)
        model.eval()
        train_embeddings = _encode(model, "qwen", train_rows, 6)
        valid_embeddings = _encode(model, "qwen", valid_rows, 6)
        probabilities = predict_head(
            train_rows,
            train_embeddings,
            valid_rows,
            valid_embeddings,
            labels,
            new_ids,
            seed,
        )
        truth = [str(row["category"]) for row in valid_rows]
        metrics = topk_metrics(truth, probabilities, labels)
        payload = {
            "index": index,
            "repeat": int(fold["repeat"]),
            "fold": int(fold["fold"]),
            "metrics": metrics,
            "train": train_stats,
            "truth": truth,
            "probabilities": probabilities.tolist(),
            "lockbox_accessed": False,
        }
        output.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        print(f"DONE historical fold={index + 1}/{len(folds)} metrics={metrics}", flush=True)
        _free_model(model)

    results = [
        json.loads((RUN_DIR / f"fold_{index:02d}.json").read_text(encoding="utf-8"))
        for index in range(len(folds))
    ]
    fold_mean = {
        key: float(np.mean([item["metrics"][key] for item in results]))
        for key in ("top1", "top3", "macro_f1_present_truth")
    }
    pooled_truth: list[str] = []
    pooled_probabilities = []
    for item in results:
        pooled_truth.extend(item["truth"])
        pooled_probabilities.append(np.asarray(item["probabilities"], dtype=float))
    pooled = topk_metrics(pooled_truth, np.vstack(pooled_probabilities), labels)

    current = joblib.load(ROOT / "models/v5/category_qwen4b_lite.joblib")["metadata"].get(
        "historical_development_stability", {}
    )
    summary = {
        "method": "12 frozen V5 DEVELOPMENT folds; 66 customer rows train-only; fixed PEFT+head recipe; no lockbox",
        "config": CONFIG.__dict__,
        "head": HEAD,
        "fold_mean": fold_mean,
        "pooled": pooled,
        "current_production_reference": current,
        "folds": [
            {k: item[k] for k in ("index", "repeat", "fold", "metrics", "train")}
            for item in results
        ],
        "lockbox_accessed": False,
    }
    (RUN_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("HISTORICAL_SUMMARY " + json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
