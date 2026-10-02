from __future__ import annotations

import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import LinearSVC

ROOT = Path(__file__).resolve().parents[1]
for import_path in (ROOT, ROOT / "backend"):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from app.ml.lite_v3_runtime import LiteV3DistilledCategoryPipeline
from app.ml.v5_runtime import (
    DecisionSoftmaxClassifier,
    QwenV5CategoryPipeline,
    build_v5_structured_features,
    specialist_text,
)
from scripts.finalize_customer_adaptation import (
    lite_text_features,
    load_all,
    normalize,
    repeated_knn_training,
    weighted_centroids,
)
from scripts.v5_experiment import EXTENDED_INSTRUCTION_REGISTRY


OUT = ROOT / "models/customer_2026-09-21"
QWEN_EMBEDDINGS = OUT / "qwen_final_embeddings.npz"
MINILM_EMBEDDINGS = OUT / "minilm_final_embeddings.npz"
SEED = 20260921


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def aligned_embeddings(path: Path, rows: list[dict]) -> np.ndarray:
    payload = np.load(path)
    ids = [str(value) for value in payload["request_ids"]]
    expected = [str(row["request_id"]) for row in rows]
    if ids != expected:
        raise RuntimeError(f"Embedding alignment failed: {path}")
    return normalize(payload["embeddings"])


def build_qwen(rows: list[dict], old_count: int, embeddings: np.ndarray) -> dict:
    labels = sorted({str(row["category"]) for row in rows})
    targets = [str(row["category"]) for row in rows]
    weights = np.concatenate([np.ones(old_count), np.full(len(rows) - old_count, 24.0)])
    encoder = OneHotEncoder(handle_unknown="ignore", min_frequency=2, sparse_output=False)
    metadata = np.asarray(encoder.fit_transform(build_v5_structured_features(rows)), dtype=float)
    train_x = np.concatenate([embeddings, metadata * 0.75], axis=1)
    estimator = LinearSVC(C=1.0, class_weight="balanced", max_iter=10000, random_state=SEED)
    estimator.fit(train_x, targets, sample_weight=weights)
    centroids = weighted_centroids(embeddings, targets, weights, labels)
    knn_x, knn_y = repeated_knn_training(embeddings, targets, weights)
    knn = KNeighborsClassifier(
        n_neighbors=7,
        weights="distance",
        metric="cosine",
        algorithm="brute",
    ).fit(knn_x, knn_y)
    pipeline = QwenV5CategoryPipeline(
        labels=labels,
        model_id="Qwen/Qwen3-Embedding-4B",
        model_revision="5cf2132abc99cad020ac570b19d031efec650f2b",
        instruction=EXTENDED_INSTRUCTION_REGISTRY["posttech_tight_c"],
        max_length=256,
        embedding_dim=2560,
        metadata_encoder=encoder,
        supervised_model=DecisionSoftmaxClassifier(estimator, temperature=1.0),
        prototype_centroids=centroids,
        knn_model=knn,
        metadata_scale=0.75,
        prototype_temperature=0.08,
        batch_size=6,
        blend_weights=(0.95, 0.025, 0.025),
        adapter_dir="models/customer_2026-09-21/qwen_r16_mnrl_s80_adapter",
    )
    historical = json.loads(
        (ROOT / "outputs/customer_gpu_finetune_2026-09-21/historical_guarded_r16_mnrl_s80/summary.json").read_text()
    )
    customer = json.loads(
        (ROOT / "outputs/customer_gpu_finetune_2026-09-21/qwen_finetuned_head_sweep.json").read_text()
    )
    metadata_payload = {
        "version": "v5.6-qwen4b-peft-customer-final",
        "trained_at": datetime.now(UTC).isoformat(),
        "training_record_count": len(rows),
        "customer_weight": 24.0,
        "blend": {"supervised": 0.95, "prototype": 0.025, "knn": 0.025},
        "customer_oof": customer.get("guarded_winner") or customer.get("winner"),
        "historical_development": historical.get("fold_mean"),
        "lockbox_accessed": False,
    }
    bundle = {
        "pipeline": pipeline,
        "threshold": 0.0,
        "margin_threshold": 0.0,
        "input_contract": "registration_mapping",
        "metadata": metadata_payload,
    }
    output = OUT / "category_qwen_finetuned.joblib"
    joblib.dump(bundle, output, compress=3)
    metadata_payload["artifact_sha256"] = sha256(output)
    (OUT / "category_qwen_finetuned.json").write_text(
        json.dumps(metadata_payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return metadata_payload


def build_lite(
    rows: list[dict],
    old_count: int,
    minilm: np.ndarray,
    teacher_qwen: np.ndarray,
) -> dict:
    labels = sorted({str(row["category"]) for row in rows})
    targets = [str(row["category"]) for row in rows]
    weights = np.concatenate([np.ones(old_count), np.full(len(rows) - old_count, 4.0)])
    projector = Ridge(alpha=1.0, fit_intercept=False)
    projector.fit(minilm, teacher_qwen, sample_weight=weights)
    projected = normalize(projector.predict(minilm))
    vectorizer = lite_text_features()
    text_matrix = vectorizer.fit_transform([specialist_text(row) for row in rows])
    metadata_encoder = OneHotEncoder(handle_unknown="ignore", min_frequency=2, sparse_output=True)
    metadata_matrix = metadata_encoder.fit_transform(build_v5_structured_features(rows))
    matrix = sparse.hstack(
        [text_matrix, metadata_matrix * 0.35, sparse.csr_matrix(projected)], format="csr"
    )
    classifier = LinearSVC(C=0.5, class_weight="balanced", max_iter=10000, random_state=SEED)
    classifier.fit(matrix, targets, sample_weight=weights)

    specialist_vectorizer = FeatureUnion([
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, max_features=60000, sublinear_tf=True)),
        ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=30000, sublinear_tf=True)),
    ])
    specialist_matrix = specialist_vectorizer.fit_transform([specialist_text(row) for row in rows])
    specialist_model = LinearSVC(C=1.0, class_weight="balanced", max_iter=10000, random_state=SEED)
    specialist_model.fit(specialist_matrix, targets, sample_weight=weights)
    current = joblib.load(ROOT / "models/v5/category_lite_v3.joblib")
    current_pipeline = current["pipeline"]
    pipeline = LiteV3DistilledCategoryPipeline(
        labels=labels,
        vectorizer=vectorizer,
        metadata_encoder=metadata_encoder,
        classifier=classifier,
        metadata_scale=0.35,
        temperature=0.173915,
        text_mode="specialist",
        projector=projector,
        minilm_model_id="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        local_model_dir="models/customer_2026-09-21/minilm_supcon_s120",
        max_seq_length=256,
        batch_size=16,
        specialist_vectorizer=specialist_vectorizer,
        specialist_model=specialist_model,
        specialist_pairs=[list(pair) for pair in getattr(current_pipeline, "specialist_pairs", set())],
        specialist_threshold=float(getattr(current_pipeline, "specialist_threshold", 0.0)),
    )
    historical = json.loads(
        (ROOT / "outputs/customer_gpu_finetune_2026-09-21/minilm_historical_supcon_s120.json").read_text()
    )
    customer = json.loads(
        (ROOT / "outputs/customer_gpu_finetune_2026-09-21/minilm_sweep.json").read_text()
    )
    metadata_payload = {
        "version": "v5.6-lite-minilm-finetuned-customer-final",
        "trained_at": datetime.now(UTC).isoformat(),
        "training_record_count": len(rows),
        "customer_weight": 4.0,
        "customer_oof": customer.get("winner", {}).get("metrics"),
        "historical_development": historical.get("fold_mean"),
        "lockbox_accessed": False,
    }
    bundle = {
        "pipeline": pipeline,
        "threshold": 0.36,
        "margin_threshold": 0.0,
        "input_contract": "registration_mapping",
        "metadata": metadata_payload,
    }
    output = OUT / "category_lite_finetuned.joblib"
    joblib.dump(bundle, output, compress=3)
    metadata_payload["artifact_sha256"] = sha256(output)
    (OUT / "category_lite_finetuned.json").write_text(
        json.dumps(metadata_payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return metadata_payload


def main() -> None:
    old_rows, new_rows, _old_minilm, _new_minilm, old_qwen, new_qwen = load_all()
    rows = old_rows + new_rows
    qwen = aligned_embeddings(QWEN_EMBEDDINGS, rows)
    minilm = aligned_embeddings(MINILM_EMBEDDINGS, rows)
    teacher = normalize(np.vstack([old_qwen, new_qwen]))
    payload = {
        "qwen": build_qwen(rows, len(old_rows), qwen),
        "lite": build_lite(rows, len(old_rows), minilm, teacher),
    }
    (OUT / "final_heads_manifest.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
