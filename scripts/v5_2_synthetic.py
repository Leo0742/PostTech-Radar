from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import LinearSVC


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.v5_dataset import DEFAULT_DATASET, load_v5_rows
from scripts.v5_experiment import (
    DEFAULT_CONTRACT,
    DEFAULT_PROTOCOL,
    labels_for_view,
    load_json,
    operator_metrics,
    truncate_embeddings,
)
from scripts.v5_gpu_runner import (
    DiskEmbeddingCache,
    SentenceTransformerBackend,
    _align_probabilities,
    _apply_instruction,
    _knn_probabilities,
    _normalize_rows,
    _prototype_probabilities,
    build_structured_features,
)


MODEL_ID = "Qwen/Qwen3-Embedding-8B"
MODEL_REVISION = "1d8ad4ca9b3dd8059ad90a75d4983776a23d44af"
INSTRUCTION = "posttech_tight_c"
FEATURE_MODE = "separate_metadata_text"
MAX_LENGTH = 512


def _load_synthetic(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        raw = json.loads(line)
        text = " ".join(str(raw.get("description") or "").split())
        label = " ".join(str(raw.get("target_category") or "").split())
        if not text or not label:
            continue
        rows.append(
            {
                "request_id": f"synthetic-v5-2-{raw.get('sample_sha256') or len(rows)}",
                "description": text,
                "category": label,
                "hard_negative_confusion_target": raw.get("hard_negative_confusion_target"),
                "source_weight": float(raw.get("synthetic_sample_weight") or 0.2),
                "quality_filters_passed": list(raw.get("quality_filters_passed") or []),
            }
        )
    return rows


def _mean_metrics(results: Sequence[dict[str, Any]]) -> dict[str, float]:
    keys = ("top1_accuracy", "top3_accuracy", "macro_f1", "true_label_mrr")
    return {key: round(statistics.fmean(float(item[key]) for item in results), 6) for key in keys}


def _select_synthetic(
    rows: Sequence[dict[str, Any]],
    train_support: Counter[str],
    labels: Sequence[str],
    *,
    allocation: str,
    ratio: float,
) -> list[dict[str, Any]]:
    by_label: dict[str, list[dict[str, Any]]] = defaultdict(list)
    label_set = set(labels)
    for row in rows:
        label = str(row["category"])
        if label in label_set and label != "Прочее":
            by_label[label].append(row)

    selected: list[dict[str, Any]] = []
    for label in labels:
        if label == "Прочее":
            continue
        support = int(train_support[str(label)])
        candidates = by_label.get(str(label), [])
        if not candidates:
            continue

        hard_pool = [row for row in candidates if row.get("hard_negative_confusion_target")]
        if allocation == "rare5":
            eligible = support <= 5
            pool = candidates
            cap = 12
        elif allocation == "rare20":
            eligible = support <= 20
            pool = candidates
            cap = 20
        elif allocation == "rare20_hard":
            eligible = support <= 20 or bool(hard_pool)
            pool = [*hard_pool, *[row for row in candidates if row not in hard_pool]]
            cap = 20
        else:
            raise ValueError(f"Unknown allocation: {allocation}")
        if not eligible:
            continue

        base = max(1, support)
        amount = min(len(pool), cap, max(1, int(math.ceil(base * ratio))))
        selected.extend(pool[:amount])
    return selected


def _fit_weighted_svc(
    x_train: np.ndarray,
    y_train: Sequence[str],
    weights: np.ndarray,
    x_validation: np.ndarray,
    labels: Sequence[str],
    *,
    seed: int,
) -> np.ndarray:
    counts = Counter(str(value) for value in y_train)
    min_count = min(counts.values())
    if min_count < 2:
        raise RuntimeError("Synthetic candidate has a class with fewer than two training rows")
    cv = min(3, min_count)
    estimator = LinearSVC(C=1.0, class_weight="balanced", random_state=seed, max_iter=10000)
    model = CalibratedClassifierCV(estimator, method="sigmoid", cv=cv)
    model.fit(x_train, y_train, sample_weight=weights)
    return _align_probabilities(model, x_validation, labels)


def main() -> None:
    parser = argparse.ArgumentParser(description="V5.2 real-only CV synthetic augmentation screen")
    parser.add_argument("--synthetic", type=Path, required=True)
    parser.add_argument("--dimension", type=int, default=3072, choices=(3072, 4096))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--cache-root", type=Path, default=ROOT / "artifacts/gpu_research_v5/embedding_cache")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "artifacts/gpu_research_v5/runs/V5_2_QWEN8B_QUALITY_MAX/synthetic_screen.json",
    )
    args = parser.parse_args()

    protocol = load_json(DEFAULT_PROTOCOL)
    contract = load_json(DEFAULT_CONTRACT)
    real_rows = load_v5_rows(DEFAULT_DATASET)
    synthetic_rows = _load_synthetic(args.synthetic)
    development_ids = set(str(value) for value in protocol["development_request_ids"])
    calibration_ids = set(str(value) for value in protocol["calibration_request_ids"])
    lockbox_ids = set(str(value) for value in protocol["internal_lockbox_request_ids"])
    allowed_ids = development_ids | calibration_ids

    # V5.2 selection must not touch lockbox rows.
    if allowed_ids & lockbox_ids:
        raise RuntimeError("Development/calibration IDs overlap lockbox IDs")
    real_rows = [row for row in real_rows if str(row["request_id"]) in allowed_ids]

    candidate = {
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "deployment_policy": "normal",
        "feature_mode": FEATURE_MODE,
        "instruction": INSTRUCTION,
        "embedding_dim": args.dimension,
        "max_length": MAX_LENGTH,
        "head": "calibrated_linearsvc_proto5_knn5",
    }
    backend = SentenceTransformerBackend(batch_size=args.batch_size)
    embedder = DiskEmbeddingCache(args.cache_root, backend)

    real_by_id = {str(row["request_id"]): row for row in real_rows}
    all_texts = [str(row["description"] or "") for row in real_rows] + [str(row["description"]) for row in synthetic_rows]
    embedded = np.asarray(embedder(_apply_instruction(all_texts, INSTRUCTION), candidate), dtype=float)
    embedded = _normalize_rows(truncate_embeddings(embedded, args.dimension))
    real_emb = {str(row["request_id"]): embedded[index] for index, row in enumerate(real_rows)}
    offset = len(real_rows)
    synthetic_emb = {str(row["request_id"]): embedded[offset + index] for index, row in enumerate(synthetic_rows)}

    allocations = ("rare5", "rare20", "rare20_hard")
    ratios = (0.5, 1.0, 2.0)
    sample_weights = (0.10, 0.20, 0.35)
    payload: dict[str, Any] = {
        "status": "complete",
        "selection_partition": "development repeated grouped CV only",
        "internal_lockbox_accessed": False,
        "model": candidate,
        "synthetic_source": str(args.synthetic),
        "synthetic_available": len(synthetic_rows),
        "views": {},
    }

    for view in ("top15", "full43"):
        labels = labels_for_view(contract, view)
        label_set = set(labels)
        grid: list[dict[str, Any]] = []
        for allocation in allocations:
            for ratio in ratios:
                for synthetic_weight in sample_weights:
                    fold_metrics = []
                    synthetic_counts = []
                    for fold in protocol["folds"]:
                        train_rows = [
                            real_by_id[str(value)]
                            for value in fold["train_request_ids"]
                            if str(value) in real_by_id and str(real_by_id[str(value)]["category"]) in label_set
                        ]
                        validation_rows = [
                            real_by_id[str(value)]
                            for value in fold["validation_request_ids"]
                            if str(value) in real_by_id and str(real_by_id[str(value)]["category"]) in label_set
                        ]
                        support = Counter(str(row["category"]) for row in train_rows)
                        selected = _select_synthetic(
                            synthetic_rows,
                            support,
                            labels,
                            allocation=allocation,
                            ratio=ratio,
                        )
                        synthetic_counts.append(len(selected))

                        encoder = OneHotEncoder(handle_unknown="ignore", min_frequency=2, sparse_output=False)
                        meta_train = encoder.fit_transform(build_structured_features(train_rows))
                        meta_validation = encoder.transform(build_structured_features(validation_rows))
                        x_real_text = np.asarray([real_emb[str(row["request_id"])] for row in train_rows], dtype=float)
                        x_validation_text = np.asarray([real_emb[str(row["request_id"])] for row in validation_rows], dtype=float)
                        x_synthetic_text = np.asarray([synthetic_emb[str(row["request_id"])] for row in selected], dtype=float)
                        if selected:
                            meta_synthetic = np.zeros((len(selected), meta_train.shape[1]), dtype=float)
                            x_train = np.concatenate(
                                [
                                    np.concatenate([x_real_text, meta_train], axis=1),
                                    np.concatenate([x_synthetic_text, meta_synthetic], axis=1),
                                ],
                                axis=0,
                            )
                        else:
                            x_train = np.concatenate([x_real_text, meta_train], axis=1)
                        x_validation = np.concatenate([x_validation_text, meta_validation], axis=1)
                        y_train = [str(row["category"]) for row in train_rows] + [str(row["category"]) for row in selected]
                        weights = np.asarray([1.0] * len(train_rows) + [synthetic_weight] * len(selected), dtype=float)

                        svc = _fit_weighted_svc(
                            x_train,
                            y_train,
                            weights,
                            x_validation,
                            labels,
                            seed=int(fold.get("seed", 20260917)),
                        )
                        prototype = _prototype_probabilities(
                            x_real_text,
                            [str(row["category"]) for row in train_rows],
                            x_validation_text,
                            labels,
                        )
                        knn = _knn_probabilities(
                            x_real_text,
                            [str(row["category"]) for row in train_rows],
                            x_validation_text,
                            labels,
                        )
                        probs = 0.90 * svc + 0.05 * prototype + 0.05 * knn
                        probs /= probs.sum(axis=1, keepdims=True)
                        fold_metrics.append(
                            operator_metrics([str(row["category"]) for row in validation_rows], probs, labels)
                        )

                    aggregate = _mean_metrics(fold_metrics)
                    grid.append(
                        {
                            "allocation": allocation,
                            "ratio": ratio,
                            "sample_weight": synthetic_weight,
                            "synthetic_rows_mean": round(statistics.fmean(synthetic_counts), 2),
                            "metrics": aggregate,
                        }
                    )

        grid.sort(
            key=lambda item: (
                item["metrics"]["top1_accuracy"],
                item["metrics"]["macro_f1"],
                item["metrics"]["top3_accuracy"],
            ),
            reverse=True,
        )
        payload["views"][view] = {"best": grid[0], "grid": grid}

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({view: payload["views"][view]["best"] for view in payload["views"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
