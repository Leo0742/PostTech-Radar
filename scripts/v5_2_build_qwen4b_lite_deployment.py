from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import LinearSVC

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "backend"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from app.ml.v5_runtime import QwenV5CategoryPipeline, specialist_text  # noqa: E402

from scripts.v5_dataset import DEFAULT_DATASET, load_v5_rows  # noqa: E402
from scripts.v5_experiment import (  # noqa: E402
    DEFAULT_CONTRACT,
    INSTRUCTION_REGISTRY,
    build_text,
    labels_for_view,
    load_json,
    truncate_embeddings,
)
from scripts.v5_gpu_runner import (  # noqa: E402
    DiskEmbeddingCache,
    SentenceTransformerBackend,
    _apply_instruction,
    _normalize_rows,
    build_structured_features,
)

MODEL_ID = "Qwen/Qwen3-Embedding-4B"
MODEL_REVISION = "5cf2132abc99cad020ac570b19d031efec650f2b"
INSTRUCTION_KEY = "posttech_tight_c"
FEATURE_MODE = "separate_metadata_text"
MAX_LENGTH = 512
EMBEDDING_DIM = 2560
METADATA_SCALE = 0.75
KNN_NEIGHBORS = 11
PROTOTYPE_TEMPERATURE = 0.08
SEED = 20260918


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fit_supervised(values: np.ndarray, labels: list[str]) -> tuple[Any, str]:
    minimum = min(Counter(labels).values())
    if minimum < 2:
        model = LogisticRegression(
            max_iter=3000,
            class_weight="balanced",
            C=1.0,
            random_state=SEED,
        )
        model.fit(values, labels)
        return model, "balanced_logistic_regression_c1_rare_class_fallback"
    estimator = LinearSVC(
        C=1.0,
        class_weight="balanced",
        random_state=SEED,
        max_iter=10000,
    )
    model = CalibratedClassifierCV(
        estimator=estimator,
        method="sigmoid",
        cv=min(3, minimum),
        n_jobs=-1,
    )
    model.fit(values, labels)
    return model, "calibrated_linearsvc_c1"


def _prototype_centroids(embeddings: np.ndarray, targets: list[str], labels: list[str]) -> np.ndarray:
    centroids = []
    normalized = _normalize_rows(embeddings)
    for label in labels:
        indices = [index for index, target in enumerate(targets) if target == label]
        if not indices:
            centroids.append(np.zeros(normalized.shape[1], dtype=float))
        else:
            centroids.append(_normalize_rows(np.mean(normalized[indices], axis=0, keepdims=True))[0])
    return np.asarray(centroids, dtype=float)


def _load_specialist(path: Path) -> tuple[list[list[str]], float, dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    winner = payload.get("winner", {})
    expected = {
        "instruction": INSTRUCTION_KEY,
        "c": 1.0,
        "blend": "svc90_proto05_knn05",
        "metadata_scale": METADATA_SCALE,
        "knn_neighbors": KNN_NEIGHBORS,
        "prototype_temperature": PROTOTYPE_TEMPERATURE,
    }
    for key, value in expected.items():
        if winner.get(key) != value:
            raise ValueError(f"Specialist validation winner mismatch for {key}: {winner.get(key)!r} != {value!r}")
    full43 = payload.get("full43", {})
    if not full43.get("enabled"):
        raise ValueError("Leakage-safe Full43 specialist validation is not enabled")
    pairs = [[str(left), str(right)] for left, right in full43.get("deployment_pairs", [])]
    if not pairs:
        raise ValueError("No validated Full43 specialist pairs found")
    threshold = float(full43.get("deployment_threshold", 0.0))
    return pairs, threshold, full43


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the frozen V5.2 Qwen3-Embedding-4B Lite bundle")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "models/v5/category_qwen4b_lite.joblib",
    )
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=ROOT / "artifacts/gpu_research_v5/embedding_cache",
    )
    parser.add_argument(
        "--specialist-validation",
        type=Path,
        default=ROOT / "outputs/qwen4b_lite/specialist_validation.json",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    contract = load_json(DEFAULT_CONTRACT)
    labels = labels_for_view(contract, "full43")
    label_set = set(labels)
    real_rows = [row for row in load_v5_rows(DEFAULT_DATASET) if str(row["category"]) in label_set]
    real_targets = [str(row["category"]) for row in real_rows]
    specialist_pairs, specialist_threshold, specialist_validation = _load_specialist(args.specialist_validation)

    candidate = {
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "feature_mode": FEATURE_MODE,
        "instruction": INSTRUCTION_KEY,
        "embedding_dim": EMBEDDING_DIM,
        "max_length": MAX_LENGTH,
        "head": "calibrated_linearsvc_proto5_knn5",
    }
    embedder = DiskEmbeddingCache(args.cache_root, SentenceTransformerBackend(batch_size=args.batch_size))
    texts = [build_text(row, feature_mode=FEATURE_MODE) for row in real_rows]
    embeddings = np.asarray(embedder(_apply_instruction(texts, INSTRUCTION_KEY), candidate), dtype=float)
    embeddings = _normalize_rows(truncate_embeddings(embeddings, EMBEDDING_DIM))

    metadata_encoder = OneHotEncoder(handle_unknown="ignore", min_frequency=2, sparse_output=False)
    metadata = np.asarray(metadata_encoder.fit_transform(build_structured_features(real_rows)), dtype=float)
    supervised_values = np.concatenate([embeddings, metadata * METADATA_SCALE], axis=1)
    supervised_model, supervised_head = _fit_supervised(supervised_values, real_targets)

    knn_model = KNeighborsClassifier(
        n_neighbors=max(1, min(KNN_NEIGHBORS, len(real_targets))),
        weights="distance",
        metric="cosine",
        algorithm="brute",
    ).fit(embeddings, real_targets)
    centroids = _prototype_centroids(embeddings, real_targets, labels)

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
    specialist_x = specialist_vectorizer.fit_transform([specialist_text(row) for row in real_rows])
    specialist_model = LinearSVC(
        C=1.0,
        class_weight="balanced",
        random_state=SEED,
        max_iter=10000,
    )
    specialist_model.fit(specialist_x, real_targets)

    pipeline = QwenV5CategoryPipeline(
        labels=labels,
        model_id=MODEL_ID,
        model_revision=MODEL_REVISION,
        instruction=INSTRUCTION_REGISTRY[INSTRUCTION_KEY],
        max_length=MAX_LENGTH,
        embedding_dim=EMBEDDING_DIM,
        metadata_encoder=metadata_encoder,
        supervised_model=supervised_model,
        prototype_centroids=centroids,
        knn_model=knn_model,
        specialist_vectorizer=specialist_vectorizer,
        specialist_model=specialist_model,
        specialist_pairs=specialist_pairs,
        specialist_threshold=specialist_threshold,
        metadata_scale=METADATA_SCALE,
        prototype_temperature=PROTOTYPE_TEMPERATURE,
        batch_size=args.batch_size,
    )
    metadata_payload = {
        "version": "v5.2-qwen4b-lite",
        "profile": "lite",
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "view": "full43",
        "feature_mode": FEATURE_MODE,
        "instruction": INSTRUCTION_KEY,
        "embedding_dim": EMBEDDING_DIM,
        "max_length": MAX_LENGTH,
        "metadata_scale": METADATA_SCALE,
        "supervised_head": supervised_head,
        "blend": {"supervised": 0.90, "prototype": 0.05, "knn": 0.05},
        "knn_neighbors": KNN_NEIGHBORS,
        "prototype_temperature": PROTOTYPE_TEMPERATURE,
        "specialist": "leakage_safe_tfidf_full43_stable_pairs",
        "specialist_pairs": len(specialist_pairs),
        "specialist_threshold": specialist_threshold,
        "development_metrics": specialist_validation.get("cross_repeat_metrics", {}),
        "development_base_metrics": specialist_validation.get("base_metrics", {}),
        "real_training_rows": len(real_rows),
        "synthetic_training_rows": 0,
        "dataset_sha256": _sha256(DEFAULT_DATASET),
        "lockbox_accessed": False,
    }
    bundle = {
        "pipeline": pipeline,
        "top15": labels_for_view(contract, "top15"),
        "threshold": 0.0,
        "margin_threshold": 0.0,
        "input_contract": "registration_mapping",
        "metadata": metadata_payload,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, args.output, compress=3)
    metadata_path = args.output.with_suffix(".json")
    metadata_path.write_text(json.dumps(metadata_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "artifact_sha256": _sha256(args.output),
                "artifact_bytes": args.output.stat().st_size,
                **metadata_payload,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
