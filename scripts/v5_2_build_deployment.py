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
from scripts.v5_2_synthetic import _load_synthetic, _select_synthetic  # noqa: E402
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


MODEL_ID = "Qwen/Qwen3-Embedding-8B"
MODEL_REVISION = "1d8ad4ca9b3dd8059ad90a75d4983776a23d44af"
INSTRUCTION_KEY = "posttech_tight_c"
FEATURE_MODE = "separate_metadata_text"
MAX_LENGTH = 512
EMBEDDING_DIM = 3072
SEED = 20260917
FULL43_SPECIALIST_PAIRS = (
    ("Ошибки загрузки страницы/зависания", "Формирование и закрытие емкостей"),
    ("Проблема с push/sms/email", "Проблема с авторизацией/ЭЗП/Бонусами/Доверенностями"),
    ("Проблема с QR-код(подключение/отключение)", "Проблема с авторизацией/ЭЗП/Бонусами/Доверенностями"),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fit_supervised(
    values: np.ndarray,
    labels: list[str],
    weights: np.ndarray,
) -> Any:
    minimum = min(Counter(labels).values())
    if minimum < 2:
        model = LogisticRegression(max_iter=3000, class_weight="balanced", C=1.0, random_state=SEED)
        model.fit(values, labels, sample_weight=weights)
        return model
    estimator = LinearSVC(C=1.0, class_weight="balanced", random_state=SEED, max_iter=10000)
    model = CalibratedClassifierCV(estimator=estimator, method="sigmoid", cv=min(3, minimum))
    model.fit(values, labels, sample_weight=weights)
    return model


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the frozen V5.2 Qwen8B deployment category bundle")
    parser.add_argument("--output", type=Path, default=ROOT / "models/v5/category.joblib")
    parser.add_argument("--cache-root", type=Path, default=ROOT / "artifacts/gpu_research_v5/embedding_cache")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--synthetic", type=Path)
    parser.add_argument("--synthetic-allocation", choices=("rare5", "rare20", "rare20_hard"))
    parser.add_argument("--synthetic-ratio", type=float)
    parser.add_argument("--synthetic-weight", type=float)
    args = parser.parse_args()

    contract = load_json(DEFAULT_CONTRACT)
    labels = labels_for_view(contract, "full43")
    label_set = set(labels)
    real_rows = [row for row in load_v5_rows(DEFAULT_DATASET) if str(row["category"]) in label_set]
    real_targets = [str(row["category"]) for row in real_rows]

    selected_synthetic: list[dict[str, Any]] = []
    synthetic_weight = 0.0
    if args.synthetic is not None:
        if args.synthetic_allocation is None or args.synthetic_ratio is None or args.synthetic_weight is None:
            raise ValueError("Synthetic deployment requires allocation, ratio, and weight")
        synthetic_rows = _load_synthetic(args.synthetic)
        selected_synthetic = _select_synthetic(
            synthetic_rows,
            Counter(real_targets),
            labels,
            allocation=args.synthetic_allocation,
            ratio=float(args.synthetic_ratio),
        )
        synthetic_weight = float(args.synthetic_weight)

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
    embedding_rows = [*real_rows, *selected_synthetic]
    texts = [build_text(row, feature_mode=FEATURE_MODE) for row in embedding_rows]
    embeddings = np.asarray(embedder(_apply_instruction(texts, INSTRUCTION_KEY), candidate), dtype=float)
    embeddings = _normalize_rows(truncate_embeddings(embeddings, EMBEDDING_DIM))
    real_embeddings = embeddings[: len(real_rows)]

    metadata_encoder = OneHotEncoder(handle_unknown="ignore", min_frequency=2, sparse_output=False)
    real_metadata = metadata_encoder.fit_transform(build_structured_features(real_rows))
    real_supervised = np.concatenate([real_embeddings, real_metadata], axis=1)
    supervised_values = real_supervised
    supervised_targets = list(real_targets)
    weights = np.ones(len(real_rows), dtype=float)
    if selected_synthetic:
        synthetic_embeddings = embeddings[len(real_rows) :]
        synthetic_metadata = np.zeros((len(selected_synthetic), real_metadata.shape[1]), dtype=float)
        supervised_values = np.concatenate(
            [real_supervised, np.concatenate([synthetic_embeddings, synthetic_metadata], axis=1)],
            axis=0,
        )
        supervised_targets.extend(str(row["category"]) for row in selected_synthetic)
        weights = np.concatenate([weights, np.full(len(selected_synthetic), synthetic_weight, dtype=float)])

    supervised_model = _fit_supervised(supervised_values, supervised_targets, weights)
    knn_model = KNeighborsClassifier(
        n_neighbors=max(1, min(9, len(real_targets))),
        weights="distance",
        metric="cosine",
        algorithm="brute",
    ).fit(real_embeddings, real_targets)
    centroids = _prototype_centroids(real_embeddings, real_targets, labels)

    specialist_vectorizer = FeatureUnion([
        (
            "char",
            TfidfVectorizer(
                analyzer="char_wb", ngram_range=(3, 5), min_df=2, max_features=60000, sublinear_tf=True
            ),
        ),
        (
            "word",
            TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=30000, sublinear_tf=True),
        ),
    ])
    specialist_x = specialist_vectorizer.fit_transform([specialist_text(row) for row in real_rows])
    specialist_model = LinearSVC(C=1.0, class_weight="balanced", random_state=20260918, max_iter=10000)
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
        specialist_pairs=FULL43_SPECIALIST_PAIRS,
        specialist_threshold=0.15,
        batch_size=args.batch_size,
    )
    bundle = {
        "pipeline": pipeline,
        "top15": labels_for_view(contract, "top15"),
        "threshold": 0.0,
        "margin_threshold": 0.0,
        "input_contract": "registration_mapping",
        "metadata": {
            "version": "v5.2",
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "view": "full43",
            "feature_mode": FEATURE_MODE,
            "instruction": INSTRUCTION_KEY,
            "embedding_dim": EMBEDDING_DIM,
            "max_length": MAX_LENGTH,
            "head": "calibrated_linearsvc+prototype_0.05+knn_0.05",
            "specialist": "safe_tfidf_full43_stable_pairs_c1_threshold_0.15",
            "real_training_rows": len(real_rows),
            "synthetic_training_rows": len(selected_synthetic),
            "synthetic_weight": synthetic_weight if selected_synthetic else None,
            "dataset_sha256": _sha256(DEFAULT_DATASET),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, args.output, compress=3)
    metadata_path = args.output.with_suffix(".json")
    metadata_path.write_text(json.dumps(bundle["metadata"], ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output), **bundle["metadata"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
