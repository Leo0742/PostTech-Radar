from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import joblib
import numpy as np
from scipy import sparse
from sklearn.metrics import f1_score
from sklearn.svm import LinearSVC


ROOT = Path(__file__).resolve().parents[1]
for import_path in (ROOT, ROOT / "backend"):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from scripts.lite_v2_feature_tune import Recipe, make_matrices  # noqa: E402
from scripts.lite_v2_sprint import aligned_probabilities  # noqa: E402
from scripts.v5_dataset import load_v5_rows  # noqa: E402


OUTPUT_DIR = ROOT / "outputs/customer_eval_2026-09-21"


def _normalize(values: np.ndarray) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return np.divide(matrix, norms, out=np.zeros_like(matrix), where=norms > 0)


def main() -> None:
    old_rows = load_v5_rows()
    new_rows = json.loads((OUTPUT_DIR / "new_rows.json").read_text(encoding="utf-8"))
    for row in new_rows:
        row["category"] = row["truth_category"]
        row["routing_target"] = row["truth_route"]

    present_labels = sorted({row["category"] for row in new_rows})
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(new_rows):
        grouped[row["category"]].append(index)
    fold_of: dict[int, int] = {}
    for _category, indices in sorted(grouped.items()):
        for position, index in enumerate(sorted(indices, key=lambda item: new_rows[item]["request_id"])):
            fold_of[index] = position % 2

    old_student = np.load(ROOT / "outputs/lite_v3/minilm_256.npy").astype(np.float32)
    new_student = np.load(OUTPUT_DIR / "minilm_new.npy").astype(np.float32)
    deployed_bundle = joblib.load(ROOT / "models/v5/category_lite_v3.joblib")
    projector = deployed_bundle["pipeline"].projector
    old_projected = _normalize(projector.predict(old_student))
    new_projected = _normalize(projector.predict(new_student))
    old_index = {row["request_id"]: index for index, row in enumerate(old_rows)}

    configs = [
        ("fused", Recipe("word11", (1, 1), (3, 5)), 0.5, 4.0),
        ("fused", Recipe("char25", (1, 2), (2, 5)), 0.5, 8.0),
        ("text", Recipe("char25", (1, 2), (2, 5)), 0.5, 8.0),
    ]
    results: list[dict[str, object]] = []
    for family, recipe, c_value, customer_weight in configs:
        predictions: list[dict[str, object]] = []
        for fold in (0, 1):
            train_new_indices = [index for index in range(len(new_rows)) if fold_of[index] != fold]
            validation_indices = [index for index in range(len(new_rows)) if fold_of[index] == fold]
            validation_services = sorted({new_rows[index].get("service") for index in validation_indices})
            for service in validation_services:
                local_validation_indices = [
                    index for index in validation_indices if new_rows[index].get("service") == service
                ]
                local_old = [row for row in old_rows if row.get("service") == service]
                local_new_indices = [
                    index for index in train_new_indices if new_rows[index].get("service") == service
                ]
                train_rows = local_old + [new_rows[index] for index in local_new_indices]
                validation_rows = [new_rows[index] for index in local_validation_indices]
                y_train = [row["category"] for row in train_rows]
                local_labels = sorted(set(y_train))
                if len(local_labels) < 2:
                    raise RuntimeError(f"Insufficient labels for service {service!r}")

                x_train, x_validation = make_matrices(train_rows, validation_rows, recipe)
                if family == "fused":
                    local_old_projected = np.vstack(
                        [old_projected[old_index[row["request_id"]]] for row in local_old]
                    )
                    if local_new_indices:
                        train_projected = np.vstack(
                            [local_old_projected, new_projected[local_new_indices]]
                        )
                    else:
                        train_projected = local_old_projected
                    x_train = sparse.hstack(
                        [x_train, sparse.csr_matrix(train_projected)], format="csr"
                    )
                    x_validation = sparse.hstack(
                        [x_validation, sparse.csr_matrix(new_projected[local_validation_indices])],
                        format="csr",
                    )

                classifier = LinearSVC(
                    C=c_value,
                    class_weight="balanced",
                    max_iter=10000,
                    random_state=20260921 + fold,
                )
                sample_weight = np.ones(len(train_rows), dtype=float)
                sample_weight[len(local_old) :] = customer_weight
                classifier.fit(x_train, y_train, sample_weight=sample_weight)
                probabilities = aligned_probabilities(
                    classifier, x_validation, local_labels, "svc"
                )
                order = np.argsort(-probabilities, axis=1)[:, :3]
                for local_index, new_index in enumerate(local_validation_indices):
                    predictions.append(
                        {
                            "id": new_rows[new_index]["request_id"],
                            "truth": new_rows[new_index]["category"],
                            "pred": local_labels[int(order[local_index, 0])],
                            "top3": [
                                local_labels[int(position)] for position in order[local_index]
                            ],
                        }
                    )

        truth = [str(item["truth"]) for item in predictions]
        predicted = [str(item["pred"]) for item in predictions]
        metrics = {
            "top1": float(np.mean([item["truth"] == item["pred"] for item in predictions])),
            "top3": float(np.mean([item["truth"] in item["top3"] for item in predictions])),
            "macro_f1_present_truth": float(
                f1_score(
                    truth,
                    predicted,
                    labels=present_labels,
                    average="macro",
                    zero_division=0,
                )
            ),
        }
        results.append(
            {
                "family": family,
                "recipe": recipe.name,
                "c": c_value,
                "customer_weight": customer_weight,
                "metrics": metrics,
                "rows": predictions,
            }
        )

    results.sort(
        key=lambda item: (
            item["metrics"]["top1"],
            item["metrics"]["macro_f1_present_truth"],
            item["metrics"]["top3"],
        ),
        reverse=True,
    )
    payload = {
        "method": "hard service-gated category models; same customer 2-fold OOF",
        "winner": {key: value for key, value in results[0].items() if key != "rows"},
        "results": [
            {key: value for key, value in item.items() if key != "rows"} for item in results
        ],
    }
    (OUTPUT_DIR / "service_gated_sweep.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (OUTPUT_DIR / "service_gated_winner_rows.json").write_text(
        json.dumps(results[0]["rows"], ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    print("WINNER ERRORS")
    for item in results[0]["rows"]:
        if item["truth"] != item["pred"]:
            print(item)


if __name__ == "__main__":
    main()
