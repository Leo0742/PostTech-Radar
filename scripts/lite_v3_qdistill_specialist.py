from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy import sparse
from sklearn.svm import LinearSVC

ROOT = Path(__file__).resolve().parents[1]
for import_path in (ROOT, ROOT / "backend"):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from scripts.lite_v2_feature_tune import Recipe as TextRecipe  # noqa: E402
from scripts.lite_v2_feature_tune import make_matrices  # noqa: E402
from scripts.lite_v2_sprint import aligned_probabilities, load_folds, mean_metrics  # noqa: E402
from scripts.lite_v3_local_max import (  # noqa: E402
    ensure_minilm_embeddings,
    fit_qwen_projection,
    load_qwen_teacher_embeddings,
)
from scripts.v5_2_qwen4b_lite_specialist_validate import validate_view  # noqa: E402
from scripts.v5_dataset import load_v5_rows  # noqa: E402
from scripts.v5_experiment import (  # noqa: E402
    DEFAULT_CONTRACT,
    labels_for_view,
    load_json,
    operator_metrics,
)

SEED = 20260920


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, default=ROOT / "outputs/lite_v3/minilm_256.npy")
    parser.add_argument(
        "--qwen-artifact",
        type=Path,
        default=ROOT / "models/v5/category_qwen4b_lite.joblib",
    )
    parser.add_argument("--device", default="mps")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-seq-length", type=int, default=256)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/lite_v3/qdistill_specialist.json")
    args = parser.parse_args()

    rows = load_v5_rows()
    labels = labels_for_view(load_json(DEFAULT_CONTRACT), "full43")
    student = ensure_minilm_embeddings(
        rows,
        args.cache,
        device=args.device,
        batch_size=args.batch_size,
        max_seq_length=args.max_seq_length,
    )
    teacher = load_qwen_teacher_embeddings(args.qwen_artifact, rows)
    teacher = teacher / np.maximum(np.linalg.norm(teacher, axis=1, keepdims=True), 1e-12)
    row_index = {str(row["request_id"]): index for index, row in enumerate(rows)}
    text_recipe = TextRecipe("word11", (1, 1), (3, 5))

    predictions = []
    fold_metrics = []
    started = time.perf_counter()
    for fold_number, (repeat, fold, train_rows, validation_rows) in enumerate(load_folds(rows), 1):
        print(f"qdistill+specialist fold {fold_number}/12 repeat={repeat} fold={fold}", flush=True)
        y_train = [str(row["category"]) for row in train_rows]
        y_validation = [str(row["category"]) for row in validation_rows]
        train_indices = [row_index[str(row["request_id"])] for row in train_rows]
        validation_indices = [row_index[str(row["request_id"])] for row in validation_rows]

        projected_train, projected_validation = fit_qwen_projection(
            student[train_indices],
            teacher[train_indices],
            student[validation_indices],
            alpha=1.0,
        )
        x_train, x_validation = make_matrices(train_rows, validation_rows, text_recipe)
        fused_train = sparse.hstack([x_train, sparse.csr_matrix(projected_train)], format="csr")
        fused_validation = sparse.hstack(
            [x_validation, sparse.csr_matrix(projected_validation)],
            format="csr",
        )
        model = LinearSVC(
            C=0.35,
            class_weight="balanced",
            max_iter=10000,
            random_state=SEED + repeat * 10 + fold,
        ).fit(fused_train, y_train)
        probabilities = aligned_probabilities(model, fused_validation, labels, "svc")
        fold_metrics.append(operator_metrics(y_validation, probabilities, labels))
        predictions.append(
            {
                "repeat": repeat,
                "fold": fold,
                "probabilities": probabilities,
                "fold_data": {
                    "labels": labels,
                    "train_rows": train_rows,
                    "validation_rows": validation_rows,
                    "y_train": y_train,
                    "y_validation": y_validation,
                },
            }
        )

    base_metrics = mean_metrics(fold_metrics)
    specialist = validate_view({"predictions": predictions, "metrics": base_metrics})
    payload = {
        "stage": "qwen_distilled_fusion_plus_specialist",
        "validation": "12 frozen grouped DEVELOPMENT folds; Qwen projection fit on each outer TRAIN fold; specialist selected leave-one-repeat-out",
        "lockbox_accessed": False,
        "qdistill_recipe": {
            "student_model": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            "teacher": "saved Qwen3-Embedding-4B 2560-d embeddings",
            "ridge_alpha": 1.0,
            "projection_scale": 1.0,
            "linear_svc_c": 0.35,
            "class_weight": "balanced",
        },
        "base_metrics": base_metrics,
        "specialist": specialist,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
