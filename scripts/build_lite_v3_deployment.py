from __future__ import annotations

import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import joblib
import scipy
import sklearn
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import LinearSVC

ROOT = Path(__file__).resolve().parents[1]
for import_path in (ROOT, ROOT / "backend"):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from app.ml.lite_v3_runtime import LiteV3DistilledCategoryPipeline  # noqa: E402
from app.ml.v5_runtime import build_v5_structured_features, specialist_text  # noqa: E402

from scripts.lite_v3_local_max import (  # noqa: E402
    MINILM_ID,
    _normalize_rows,
    ensure_minilm_embeddings,
    load_qwen_teacher_embeddings,
)
from scripts.v5_dataset import DEFAULT_DATASET, load_v5_rows  # noqa: E402
from scripts.v5_experiment import DEFAULT_CONTRACT, labels_for_view, load_json  # noqa: E402

SEED = 20260920


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    validation = json.loads((ROOT / "outputs/lite_v3/qdistill_specialist.json").read_text(encoding="utf-8"))
    specialist_validation = validation["specialist"]
    calibration = json.loads(
        (ROOT / "outputs/lite_v3/calibration_qdistill_specialist.json").read_text(encoding="utf-8")
    )
    rows = load_v5_rows(DEFAULT_DATASET)
    labels = labels_for_view(load_json(DEFAULT_CONTRACT), "full43")
    targets = [str(row["category"]) for row in rows]
    texts = [specialist_text(row) for row in rows]

    student = ensure_minilm_embeddings(
        rows,
        ROOT / "outputs/lite_v3/minilm_256.npy",
        device="mps",
        batch_size=32,
        max_seq_length=256,
    )
    teacher = _normalize_rows(load_qwen_teacher_embeddings(ROOT / "models/v5/category_qwen4b_lite.joblib", rows))
    projector = Ridge(alpha=1.0, fit_intercept=False)
    projector.fit(student, teacher)
    projected = _normalize_rows(projector.predict(student))

    vectorizer = FeatureUnion(
        [
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
        ]
    )
    text_matrix = vectorizer.fit_transform(texts)
    metadata_encoder = OneHotEncoder(handle_unknown="ignore", min_frequency=2, sparse_output=True)
    metadata_matrix = metadata_encoder.fit_transform(build_v5_structured_features(rows))
    matrix = sparse.hstack(
        [text_matrix, metadata_matrix * 0.35, sparse.csr_matrix(projected)],
        format="csr",
    )
    classifier = LinearSVC(
        C=0.35,
        class_weight="balanced",
        max_iter=10000,
        random_state=SEED,
    ).fit(matrix, targets)

    specialist_vectorizer = FeatureUnion(
        [
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
        ]
    )
    specialist_matrix = specialist_vectorizer.fit_transform(texts)
    specialist_model = LinearSVC(
        C=1.0,
        class_weight="balanced",
        max_iter=10000,
        random_state=SEED,
    ).fit(specialist_matrix, targets)

    pipeline = LiteV3DistilledCategoryPipeline(
        labels=labels,
        vectorizer=vectorizer,
        metadata_encoder=metadata_encoder,
        classifier=classifier,
        metadata_scale=0.35,
        temperature=float(calibration["temperature"]),
        text_mode="specialist",
        projector=projector,
        minilm_model_id=MINILM_ID,
        local_model_dir="models/encoders/paraphrase-multilingual-MiniLM-L12-v2",
        max_seq_length=256,
        batch_size=16,
        specialist_vectorizer=specialist_vectorizer,
        specialist_model=specialist_model,
        specialist_pairs=specialist_validation["deployment_pairs"],
        specialist_threshold=float(specialist_validation["deployment_threshold"]),
    )

    metadata = {
        "version": "v5.4-lite-v3-qdistill",
        "candidate_id": "lite-v3-qdistill-a1-fusion-s1-c035-specialist",
        "profile": "lite",
        "model_family": "lite_v3_qwen_distilled",
        "family": "TF-IDF word+char + metadata + Qwen-distilled MiniLM projection + LinearSVC + specialist",
        "trained_at": datetime.now(UTC).isoformat(),
        "training_record_count": len(rows),
        "labels": len(labels),
        "dataset_sha256": sha256(DEFAULT_DATASET),
        "input_contract": "registration_mapping",
        "student_model": MINILM_ID,
        "student_local_dir": "models/encoders/paraphrase-multilingual-MiniLM-L12-v2",
        "teacher": "saved Qwen3-Embedding-4B 2560-d embeddings; teacher is not required at runtime",
        "projection": {"type": "Ridge", "alpha": 1.0, "fit_intercept": False, "output_dim": 2560},
        "classifier": {"type": "LinearSVC", "c": 0.35, "class_weight": "balanced"},
        "specialist": {
            "pairs": specialist_validation["deployment_pairs"],
            "threshold": specialist_validation["deployment_threshold"],
            "cross_repeat_metrics": specialist_validation["cross_repeat_metrics"],
            "corrected_errors": specialist_validation["cross_repeat_corrected_errors"],
            "introduced_errors": specialist_validation["cross_repeat_introduced_errors"],
        },
        "development_metrics": specialist_validation["cross_repeat_metrics"],
        "development_validation": specialist_validation["validation"],
        "calibration": calibration,
        "lockbox_accessed": False,
        "python": sys.version.split()[0],
        "sklearn": sklearn.__version__,
        "scipy": scipy.__version__,
    }
    bundle = {
        "pipeline": pipeline,
        "top15": labels_for_view(load_json(DEFAULT_CONTRACT), "top15"),
        "threshold": float(calibration["threshold"]),
        "margin_threshold": float(calibration["margin_threshold"]),
        "input_contract": "registration_mapping",
        "metadata": metadata,
    }
    output = ROOT / "models/v5/category_lite_v3.joblib"
    output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, output, compress=3)
    metadata["artifact_sha256"] = sha256(output)
    metadata["artifact_bytes"] = output.stat().st_size
    output.with_suffix(".json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"artifact": str(output), **metadata}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
