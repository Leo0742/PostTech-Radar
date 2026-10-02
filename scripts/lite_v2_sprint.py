from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import LinearSVC

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "backend"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from app.ml.training import registration_text  # noqa: E402
from app.ml.v5_runtime import build_v5_structured_features, specialist_text  # noqa: E402
from scripts.v5_dataset import load_v5_rows  # noqa: E402
from scripts.v5_experiment import DEFAULT_CONTRACT, labels_for_view, load_json, operator_metrics  # noqa: E402

SEED = 20260920
FOLDS_PATH = ROOT / "artifacts/gpu_research_v5/protocol/folds.json"


@dataclass(frozen=True)
class FeatureRecipe:
    name: str
    text_mode: str
    word_max: int
    char_max: int
    word_ngram: tuple[int, int]
    char_ngram: tuple[int, int]
    structured: bool = False
    metadata_scale: float = 1.0


@dataclass(frozen=True)
class ClassifierRecipe:
    name: str
    family: str
    c: float
    class_weight: str | None


def row_text(row: dict[str, Any], mode: str) -> str:
    if mode == "description":
        return " ".join(str(row.get("description") or "").split())
    if mode == "specialist":
        return specialist_text(row)
    if mode == "combined":
        return registration_text(row, "combined")
    raise ValueError(mode)


def make_vectorizer(recipe: FeatureRecipe) -> FeatureUnion:
    return FeatureUnion([
        (
            "word",
            TfidfVectorizer(
                lowercase=True,
                ngram_range=recipe.word_ngram,
                min_df=2,
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
                min_df=2,
                max_features=recipe.char_max,
                sublinear_tf=True,
            ),
        ),
    ])


def matrices(train_rows: list[dict[str, Any]], val_rows: list[dict[str, Any]], recipe: FeatureRecipe):
    vectorizer = make_vectorizer(recipe)
    x_train = vectorizer.fit_transform([row_text(row, recipe.text_mode) for row in train_rows])
    x_val = vectorizer.transform([row_text(row, recipe.text_mode) for row in val_rows])
    if not recipe.structured:
        return x_train, x_val

    encoder = OneHotEncoder(handle_unknown="ignore", min_frequency=2, sparse_output=True)
    m_train = encoder.fit_transform(build_v5_structured_features(train_rows))
    m_val = encoder.transform(build_v5_structured_features(val_rows))
    x_train = sparse.hstack([x_train, m_train * recipe.metadata_scale], format="csr")
    x_val = sparse.hstack([x_val, m_val * recipe.metadata_scale], format="csr")
    return x_train, x_val


def classifier(recipe: ClassifierRecipe):
    if recipe.family == "lr":
        return LogisticRegression(
            C=recipe.c,
            class_weight=recipe.class_weight,
            max_iter=2500,
            random_state=SEED,
            solver="lbfgs",
        )
    if recipe.family == "svc":
        return LinearSVC(
            C=recipe.c,
            class_weight=recipe.class_weight,
            max_iter=10000,
            random_state=SEED,
        )
    raise ValueError(recipe.family)


def softmax(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    values -= np.max(values, axis=1, keepdims=True)
    exp = np.exp(values)
    return exp / exp.sum(axis=1, keepdims=True)


def aligned_probabilities(model: Any, x: Any, labels: list[str], family: str) -> np.ndarray:
    if family == "lr":
        local = np.asarray(model.predict_proba(x), dtype=float)
    else:
        local = softmax(np.asarray(model.decision_function(x), dtype=float))
    result = np.zeros((x.shape[0], len(labels)), dtype=float)
    target = {label: index for index, label in enumerate(labels)}
    for source, label in enumerate(model.classes_):
        index = target.get(str(label))
        if index is not None:
            result[:, index] = local[:, source]
    sums = result.sum(axis=1, keepdims=True)
    return np.divide(result, sums, out=np.zeros_like(result), where=sums > 0)


def mean_metrics(records: list[dict[str, Any]]) -> dict[str, float]:
    keys = ("top1_accuracy", "top3_accuracy", "macro_f1", "true_label_mrr")
    return {key: round(mean(float(item[key]) for item in records), 9) for key in keys}


def quality_key(metrics: dict[str, float]) -> tuple[float, float, float, float]:
    return (
        metrics["top1_accuracy"],
        metrics["macro_f1"],
        metrics["top3_accuracy"],
        metrics["true_label_mrr"],
    )


def load_folds(rows: list[dict[str, Any]]):
    by_id = {str(row["request_id"]): row for row in rows}
    raw = json.loads(FOLDS_PATH.read_text(encoding="utf-8"))
    result = []
    for item in raw:
        train = [by_id[str(value)] for value in item["train_request_ids"]]
        validation = [by_id[str(value)] for value in item["validation_request_ids"]]
        result.append((int(item["repeat"]), int(item["fold"]), train, validation))
    return result


def coarse_features() -> list[FeatureRecipe]:
    return [
        FeatureRecipe("current48_combined", "combined", 20000, 28000, (1, 3), (2, 6)),
        FeatureRecipe("qwen90_specialist", "specialist", 30000, 60000, (1, 2), (3, 5)),
        FeatureRecipe("rich90_combined", "combined", 30000, 60000, (1, 3), (3, 5)),
        FeatureRecipe("desc90_struct075", "description", 30000, 60000, (1, 3), (3, 5), True, 0.75),
        FeatureRecipe("rich90_struct075", "combined", 30000, 60000, (1, 3), (3, 5), True, 0.75),
        FeatureRecipe("qwen90_struct075", "specialist", 30000, 60000, (1, 2), (3, 5), True, 0.75),
    ]


def feature_recipes(stage: str, winner: dict[str, Any] | None = None) -> list[FeatureRecipe]:
    if stage == "coarse" or winner is None:
        return coarse_features()
    base = next(item for item in coarse_features() if item.name == winner["feature"])
    if stage == "scale" and base.structured:
        stem = base.name.rsplit("_struct", 1)[0]
        return [
            FeatureRecipe(
                f"{stem}_struct{str(scale).replace('.', '')}",
                base.text_mode,
                base.word_max,
                base.char_max,
                base.word_ngram,
                base.char_ngram,
                True,
                scale,
            )
            for scale in (0.20, 0.35, 0.50, 0.65, 0.75, 0.90, 1.10, 1.30, 1.50)
        ]
    return [base]


def classifier_recipes(stage: str) -> list[ClassifierRecipe]:
    if stage == "coarse":
        return [
            ClassifierRecipe("lr_c2_bal", "lr", 2.0, "balanced"),
            ClassifierRecipe("svc_c08_bal", "svc", 0.8, "balanced"),
            ClassifierRecipe("svc_c08_none", "svc", 0.8, None),
        ]
    if stage == "scale":
        return [ClassifierRecipe("svc_c1.0_bal", "svc", 1.0, "balanced")]
    result = []
    for c in (0.10, 0.15, 0.20, 0.25, 0.35, 0.50, 0.65, 0.80, 1.0, 1.2, 1.6, 2.0, 3.0):
        for weight in ("balanced", None):
            suffix = "bal" if weight else "none"
            result.append(ClassifierRecipe(f"svc_c{c}_{suffix}", "svc", c, weight))
    return result


def evaluate(stage: str, output: Path, winner_path: Path | None = None) -> None:
    rows = load_v5_rows()
    labels = labels_for_view(load_json(DEFAULT_CONTRACT), "full43")
    folds = load_folds(rows)
    previous = None
    if winner_path and winner_path.exists():
        previous = json.loads(winner_path.read_text(encoding="utf-8")).get("winner")
    features = feature_recipes(stage, previous)
    classifiers = classifier_recipes(stage)
    scores = {(f.name, c.name): [] for f in features for c in classifiers}
    timings = {(f.name, c.name): 0.0 for f in features for c in classifiers}

    started_all = time.perf_counter()
    for fold_index, (repeat, fold, train_rows, val_rows) in enumerate(folds, 1):
        y_train = [str(row["category"]) for row in train_rows]
        y_val = [str(row["category"]) for row in val_rows]
        print(f"fold {fold_index}/{len(folds)} repeat={repeat} fold={fold}", flush=True)
        for feat in features:
            build_started = time.perf_counter()
            x_train, x_val = matrices(train_rows, val_rows, feat)
            build_elapsed = time.perf_counter() - build_started
            for clf_recipe in classifiers:
                key = (feat.name, clf_recipe.name)
                model = classifier(clf_recipe)
                train_started = time.perf_counter()
                model.fit(x_train, y_train)
                probabilities = aligned_probabilities(model, x_val, labels, clf_recipe.family)
                timings[key] += (time.perf_counter() - train_started) + build_elapsed / len(classifiers)
                scores[key].append(operator_metrics(y_val, probabilities, labels))

    results = []
    for feat in features:
        for clf_recipe in classifiers:
            key = (feat.name, clf_recipe.name)
            metrics = mean_metrics(scores[key])
            results.append(
                {
                    "feature": feat.name,
                    "feature_recipe": asdict(feat),
                    "classifier": clf_recipe.name,
                    "classifier_recipe": asdict(clf_recipe),
                    "metrics": metrics,
                    "elapsed_seconds": round(timings[key], 3),
                }
            )
    results.sort(key=lambda item: quality_key(item["metrics"]), reverse=True)
    payload = {
        "stage": stage,
        "folds": len(folds),
        "rows": len(rows),
        "labels": len(labels),
        "elapsed_seconds": round(time.perf_counter() - started_all, 3),
        "winner": results[0],
        "results": results,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"winner": results[0], "top5": results[:5], "elapsed_seconds": payload["elapsed_seconds"]}, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("coarse", "refine", "scale"), default="coarse")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--winner-from", type=Path)
    args = parser.parse_args()
    evaluate(args.stage, args.output, args.winner_from)


if __name__ == "__main__":
    main()
