from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import OneHotEncoder

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "backend"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from app.ml.v5_runtime import build_v5_structured_features, specialist_text  # noqa: E402
from scripts.lite_v2_sprint import (  # noqa: E402
    ClassifierRecipe,
    aligned_probabilities,
    classifier,
    load_folds,
)
from scripts.v5_dataset import load_v5_rows  # noqa: E402
from scripts.v5_experiment import DEFAULT_CONTRACT, labels_for_view, load_json, operator_metrics  # noqa: E402


@dataclass(frozen=True)
class Recipe:
    name: str
    word_ngram: tuple[int, int]
    char_ngram: tuple[int, int]
    word_min_df: int = 2
    char_min_df: int = 2
    word_weight: float = 1.0
    char_weight: float = 1.0
    word_max: int = 30000
    char_max: int = 60000


RECIPES = [
    Recipe("base", (1, 2), (3, 5)),
    Recipe("word13", (1, 3), (3, 5)),
    Recipe("word11", (1, 1), (3, 5)),
    Recipe("char25", (1, 2), (2, 5)),
    Recipe("char26", (1, 2), (2, 6)),
    Recipe("char36", (1, 2), (3, 6)),
    Recipe("word13_char25", (1, 3), (2, 5)),
    Recipe("min_df1", (1, 2), (3, 5), 1, 1),
    Recipe("word125", (1, 2), (3, 5), 2, 2, 1.25, 1.0),
    Recipe("word150", (1, 2), (3, 5), 2, 2, 1.50, 1.0),
    Recipe("char125", (1, 2), (3, 5), 2, 2, 1.0, 1.25),
    Recipe("char150", (1, 2), (3, 5), 2, 2, 1.0, 1.50),
    Recipe("bigger", (1, 3), (3, 6), 2, 2, 1.0, 1.0, 50000, 100000),
]


def make_matrices(train_rows, validation_rows, recipe: Recipe):
    vectorizer = FeatureUnion(
        [
            (
                "word",
                TfidfVectorizer(
                    ngram_range=recipe.word_ngram,
                    min_df=recipe.word_min_df,
                    max_df=0.995,
                    max_features=recipe.word_max,
                    sublinear_tf=True,
                ),
            ),
            (
                "char",
                TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=recipe.char_ngram,
                    min_df=recipe.char_min_df,
                    max_features=recipe.char_max,
                    sublinear_tf=True,
                ),
            ),
        ],
        transformer_weights={"word": recipe.word_weight, "char": recipe.char_weight},
    )
    train_text = [specialist_text(row) for row in train_rows]
    validation_text = [specialist_text(row) for row in validation_rows]
    text_train = vectorizer.fit_transform(train_text)
    text_validation = vectorizer.transform(validation_text)
    encoder = OneHotEncoder(handle_unknown="ignore", min_frequency=2, sparse_output=True)
    meta_train = encoder.fit_transform(build_v5_structured_features(train_rows))
    meta_validation = encoder.transform(build_v5_structured_features(validation_rows))
    return (
        sparse.hstack([text_train, meta_train * 0.35], format="csr"),
        sparse.hstack([text_validation, meta_validation * 0.35], format="csr"),
    )


def mean_metrics(records):
    keys = ("top1_accuracy", "top3_accuracy", "macro_f1", "true_label_mrr")
    return {key: round(mean(float(item[key]) for item in records), 9) for key in keys}


def quality_key(metrics):
    return (metrics["top1_accuracy"], metrics["macro_f1"], metrics["top3_accuracy"], metrics["true_label_mrr"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/lite_v2/features.json")
    args = parser.parse_args()

    rows = load_v5_rows()
    labels = labels_for_view(load_json(DEFAULT_CONTRACT), "full43")
    model_recipe = ClassifierRecipe("svc_c1.0_bal", "svc", 1.0, "balanced")
    scores = {recipe.name: [] for recipe in RECIPES}
    timings = {recipe.name: 0.0 for recipe in RECIPES}
    started = time.perf_counter()

    for fold_index, (_repeat, _fold, train_rows, validation_rows) in enumerate(load_folds(rows), 1):
        print(f"fold {fold_index}/12", flush=True)
        y_train = [str(row["category"]) for row in train_rows]
        y_validation = [str(row["category"]) for row in validation_rows]
        for recipe in RECIPES:
            step = time.perf_counter()
            x_train, x_validation = make_matrices(train_rows, validation_rows, recipe)
            model = classifier(model_recipe)
            model.fit(x_train, y_train)
            probabilities = aligned_probabilities(model, x_validation, labels, "svc")
            timings[recipe.name] += time.perf_counter() - step
            scores[recipe.name].append(operator_metrics(y_validation, probabilities, labels))

    results = [
        {
            "recipe": asdict(recipe),
            "metrics": mean_metrics(scores[recipe.name]),
            "elapsed_seconds": round(timings[recipe.name], 3),
        }
        for recipe in RECIPES
    ]
    results.sort(key=lambda item: quality_key(item["metrics"]), reverse=True)
    payload = {
        "classifier": model_recipe.__dict__,
        "metadata_scale": 0.35,
        "folds": 12,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "winner": results[0],
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"winner": results[0], "top5": results[:5], "elapsed_seconds": payload["elapsed_seconds"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
