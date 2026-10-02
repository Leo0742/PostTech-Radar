from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "backend"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from scripts.lite_v2_sprint import (  # noqa: E402
    ClassifierRecipe,
    FeatureRecipe,
    aligned_probabilities,
    classifier,
    load_folds,
    matrices,
    mean_metrics,
)
from scripts.v5_2_qwen4b_lite_specialist_validate import validate_view  # noqa: E402
from scripts.v5_dataset import load_v5_rows  # noqa: E402
from scripts.v5_experiment import DEFAULT_CONTRACT, labels_for_view, load_json, operator_metrics  # noqa: E402


WINNER_FEATURE = FeatureRecipe(
    "qwen90_struct035",
    "specialist",
    30000,
    60000,
    (1, 2),
    (3, 5),
    True,
    0.35,
)
WINNER_CLASSIFIER = ClassifierRecipe("svc_c1.0_bal", "svc", 1.0, "balanced")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/lite_v2/specialist.json")
    args = parser.parse_args()

    rows = load_v5_rows()
    labels = labels_for_view(load_json(DEFAULT_CONTRACT), "full43")
    predictions = []
    fold_metrics = []

    for index, (repeat, fold, train_rows, validation_rows) in enumerate(load_folds(rows), 1):
        print(f"fold {index}/12 repeat={repeat} fold={fold}", flush=True)
        x_train, x_validation = matrices(train_rows, validation_rows, WINNER_FEATURE)
        y_train = [str(row["category"]) for row in train_rows]
        y_validation = [str(row["category"]) for row in validation_rows]
        model = classifier(WINNER_CLASSIFIER)
        model.fit(x_train, y_train)
        probabilities = aligned_probabilities(model, x_validation, labels, WINNER_CLASSIFIER.family)
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
        "feature_recipe": WINNER_FEATURE.__dict__,
        "classifier_recipe": WINNER_CLASSIFIER.__dict__,
        "base_metrics": base_metrics,
        "specialist": specialist,
        "lockbox_accessed": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
