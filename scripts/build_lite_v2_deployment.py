from __future__ import annotations

import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import joblib
import scipy
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import LinearSVC

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "backend"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from app.ml.lite_v2_runtime import LiteV2CategoryPipeline  # noqa: E402
from app.ml.v5_runtime import build_v5_structured_features, specialist_text  # noqa: E402

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
    specialist_calibration = ROOT / "outputs/lite_v3/calibration_specialist.json"
    calibration_path = (
        specialist_calibration
        if specialist_calibration.exists()
        else ROOT / "outputs/lite_v2/calibration.json"
    )
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    development = json.loads((ROOT / "outputs/lite_v2/micro_tune.json").read_text(encoding="utf-8"))
    specialist_validation = json.loads(
        (ROOT / "outputs/lite_v3/final_baseline_specialist.json").read_text(encoding="utf-8")
    )["specialist"]
    winner = development["winner"]
    if (winner["c"], winner["word_weight"], winner["metadata_scale"]) != (0.35, 1.0, 0.35):
        raise RuntimeError("Unexpected Lite V2 winner; review deployment recipe before building")

    rows = load_v5_rows(DEFAULT_DATASET)
    labels = labels_for_view(load_json(DEFAULT_CONTRACT), "full43")
    targets = [str(row["category"]) for row in rows]
    texts = [specialist_text(row) for row in rows]

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

    from scipy import sparse

    matrix = sparse.hstack([text_matrix, metadata_matrix * 0.35], format="csr")
    classifier = LinearSVC(
        C=0.35,
        class_weight="balanced",
        max_iter=10000,
        random_state=SEED,
    ).fit(matrix, targets)

    specialist_vectorizer = FeatureUnion(
        [
            (
                "word",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    min_df=2,
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
    specialist_matrix = specialist_vectorizer.fit_transform(texts)
    specialist_model = LinearSVC(
        C=1.0,
        class_weight="balanced",
        max_iter=10000,
        random_state=SEED,
    ).fit(specialist_matrix, targets)
    specialist_pairs = specialist_validation["deployment_pairs"]
    specialist_threshold = float(specialist_validation["deployment_threshold"])

    pipeline = LiteV2CategoryPipeline(
        labels=labels,
        vectorizer=vectorizer,
        metadata_encoder=metadata_encoder,
        classifier=classifier,
        metadata_scale=0.35,
        temperature=float(calibration["temperature"]),
        text_mode="specialist",
        specialist_vectorizer=specialist_vectorizer,
        specialist_model=specialist_model,
        specialist_pairs=specialist_pairs,
        specialist_threshold=specialist_threshold,
    )

    metadata = {
        "version": "v5.3-lite-v2-specialist",
        "candidate_id": "lite-v2-tfidf-svc-c035-specialist",
        "profile": "lite",
        "model_family": "lite_v2_tfidf_svc",
        "family": "TF-IDF word+char + metadata + LinearSVC + leakage-safe TF-IDF specialist",
        "trained_at": datetime.now(UTC).isoformat(),
        "training_record_count": len(rows),
        "labels": len(labels),
        "dataset_sha256": sha256(DEFAULT_DATASET),
        "input_contract": "registration_mapping",
        "feature_recipe": {
            "text_builder": "specialist_text",
            "word_ngram": [1, 1],
            "word_min_df": 2,
            "word_max_features": 30000,
            "char_analyzer": "char_wb",
            "char_ngram": [3, 5],
            "char_min_df": 2,
            "char_max_features": 60000,
            "structured_metadata_scale": 0.35,
        },
        "classifier": {"type": "LinearSVC", "c": 0.35, "class_weight": "balanced"},
        "specialist": {
            "type": "TF-IDF word(1,2)+char_wb(3,5) + LinearSVC",
            "c": 1.0,
            "class_weight": "balanced",
            "pairs": specialist_pairs,
            "threshold": specialist_threshold,
            "validation": specialist_validation["validation"],
            "cross_repeat_metrics": specialist_validation["cross_repeat_metrics"],
            "cross_repeat_top1_delta": specialist_validation["cross_repeat_top1_delta"],
            "cross_repeat_macro_f1_delta": specialist_validation["cross_repeat_macro_f1_delta"],
        },
        "calibration": calibration,
        "calibration_source": str(calibration_path.relative_to(ROOT)),
        "development_metrics": winner["metrics"],
        "development_validation": development["validation"],
        "lockbox_accessed": False,
        "qwen_knowledge_reused": (
            "V5/Qwen-derived specialist text representation and structured metadata schema; "
            "Qwen weights are not required by Lite inference"
        ),
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

    output = ROOT / "models/v5/category_lite_v2.joblib"
    output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, output, compress=3)
    metadata["artifact_sha256"] = sha256(output)
    metadata["artifact_bytes"] = output.stat().st_size
    output.with_suffix(".json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"artifact": str(output), **metadata}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
