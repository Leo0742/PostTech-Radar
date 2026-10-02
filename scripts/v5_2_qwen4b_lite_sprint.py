from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any

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

from app.ml.v5_runtime import specialist_text  # noqa: E402

from scripts.v5_dataset import DEFAULT_DATASET, load_v5_rows  # noqa: E402
from scripts.v5_experiment import (  # noqa: E402
    DEFAULT_CONTRACT,
    DEFAULT_PROTOCOL,
    build_text,
    labels_for_view,
    load_json,
    operator_metrics,
    truncate_embeddings,
)
from scripts.v5_gpu_runner import (  # noqa: E402
    DiskEmbeddingCache,
    SentenceTransformerBackend,
    _align_probabilities,
    _apply_instruction,
    _fold_for_candidate,
    _normalize_rows,
    build_structured_features,
)

MODEL_ID = "Qwen/Qwen3-Embedding-4B"
MODEL_REVISION = "5cf2132abc99cad020ac570b19d031efec650f2b"
FEATURE_MODE = "separate_metadata_text"
MAX_LENGTH = 512
EMBEDDING_DIM = 2560
SEED = 20260918
BASELINE_INSTRUCTION = "en_concise"
ALT_INSTRUCTION = "posttech_tight_c"

BLENDS: dict[str, tuple[float, float, float]] = {
    "svc90_proto05_knn05": (0.90, 0.05, 0.05),
    "svc95_proto025_knn025": (0.95, 0.025, 0.025),
    "svc90_proto075_knn025": (0.90, 0.075, 0.025),
    "svc90_proto025_knn075": (0.90, 0.025, 0.075),
    "svc100": (1.0, 0.0, 0.0),
}


@dataclass(frozen=True)
class Recipe:
    instruction: str = BASELINE_INSTRUCTION
    c: float = 1.0
    blend: str = "svc90_proto05_knn05"
    metadata_scale: float = 1.0
    knn_neighbors: int = 9
    prototype_temperature: float = 0.08

    def as_dict(self) -> dict[str, Any]:
        return {
            "instruction": self.instruction,
            "c": self.c,
            "blend": self.blend,
            "blend_weights": BLENDS[self.blend],
            "metadata_scale": self.metadata_scale,
            "knn_neighbors": self.knn_neighbors,
            "prototype_temperature": self.prototype_temperature,
        }


def _quality_key(metrics: Mapping[str, Any]) -> tuple[float, float, float]:
    return (
        float(metrics["top1_accuracy"]),
        float(metrics["macro_f1"]),
        float(metrics["top3_accuracy"]),
    )


def _mean_metrics(records: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    keys = ("top1_accuracy", "top3_accuracy", "macro_f1", "true_label_mrr")
    return {key: round(mean(float(record[key]) for record in records), 9) for key in keys}


def _fit_calibrated(
    x_train: np.ndarray,
    y_train: Sequence[str],
    x_validation: np.ndarray,
    labels: Sequence[str],
    *,
    c: float,
    seed: int,
) -> np.ndarray:
    minimum = min(Counter(str(value) for value in y_train).values())
    if minimum < 2:
        model = LogisticRegression(
            max_iter=3000,
            class_weight="balanced",
            C=1.0,
            random_state=seed,
        )
        model.fit(x_train, y_train)
        return _align_probabilities(model, x_validation, labels)
    estimator = LinearSVC(
        C=float(c),
        class_weight="balanced",
        random_state=seed,
        max_iter=10000,
    )
    model = CalibratedClassifierCV(
        estimator=estimator,
        method="sigmoid",
        cv=min(3, minimum),
        n_jobs=-1,
    )
    model.fit(x_train, y_train)
    return _align_probabilities(model, x_validation, labels)


def _prototype_probabilities(
    x_train: np.ndarray,
    y_train: Sequence[str],
    x_validation: np.ndarray,
    labels: Sequence[str],
    *,
    temperature: float,
) -> np.ndarray:
    train = _normalize_rows(x_train)
    validation = _normalize_rows(x_validation)
    centroids = []
    for label in labels:
        indices = [i for i, value in enumerate(y_train) if str(value) == str(label)]
        if not indices:
            centroids.append(np.zeros(train.shape[1], dtype=float))
        else:
            centroids.append(_normalize_rows(np.mean(train[indices], axis=0, keepdims=True))[0])
    logits = validation @ np.asarray(centroids, dtype=float).T
    scaled = logits / max(float(temperature), 1e-6)
    scaled -= np.max(scaled, axis=1, keepdims=True)
    exp = np.exp(scaled)
    return exp / exp.sum(axis=1, keepdims=True)


def _knn_probabilities(
    x_train: np.ndarray,
    y_train: Sequence[str],
    x_validation: np.ndarray,
    labels: Sequence[str],
    *,
    neighbors: int,
) -> np.ndarray:
    model = KNeighborsClassifier(
        n_neighbors=max(1, min(int(neighbors), len(y_train))),
        weights="distance",
        metric="cosine",
        algorithm="brute",
    )
    model.fit(x_train, y_train)
    return _align_probabilities(model, x_validation, labels)


class FoldStore:
    def __init__(
        self,
        *,
        rows: list[dict[str, Any]],
        protocol: Mapping[str, Any],
        contract: Mapping[str, Any],
        cache_root: Path,
        batch_size: int,
    ) -> None:
        self.rows = rows
        self.protocol = protocol
        self.contract = contract
        self.rows_by_id = {str(row["request_id"]): row for row in rows}
        self.embedder = DiskEmbeddingCache(cache_root, SentenceTransformerBackend(batch_size=batch_size))
        self.memory: dict[tuple[str, int, int, str], dict[str, Any]] = {}

    def get(self, view: str, repeat: int, fold_index: int, instruction: str) -> dict[str, Any]:
        key = (view, int(repeat), int(fold_index), instruction)
        if key in self.memory:
            return self.memory[key]
        labels = labels_for_view(self.contract, view)
        label_set = set(labels)
        candidate = {
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "feature_mode": FEATURE_MODE,
            "instruction": instruction,
            "embedding_dim": EMBEDDING_DIM,
            "max_length": MAX_LENGTH,
            "repeat": int(repeat),
            "fold": int(fold_index),
        }
        fold = _fold_for_candidate(self.protocol, candidate)
        train_rows = [
            self.rows_by_id[str(value)]
            for value in fold["train_request_ids"]
            if str(value) in self.rows_by_id
            and str(self.rows_by_id[str(value)]["category"]) in label_set
        ]
        validation_rows = [
            self.rows_by_id[str(value)]
            for value in fold["validation_request_ids"]
            if str(value) in self.rows_by_id
            and str(self.rows_by_id[str(value)]["category"]) in label_set
        ]
        all_rows = {str(row["request_id"]): row for row in [*train_rows, *validation_rows]}
        ordered_rows = [all_rows[value] for value in sorted(all_rows)]
        texts = [build_text(row, feature_mode=FEATURE_MODE) for row in ordered_rows]
        values = np.asarray(
            self.embedder(_apply_instruction(texts, instruction), candidate),
            dtype=float,
        )
        values = _normalize_rows(truncate_embeddings(values, EMBEDDING_DIM))
        by_id = {str(row["request_id"]): values[i] for i, row in enumerate(ordered_rows)}
        x_train_text = np.asarray([by_id[str(row["request_id"])] for row in train_rows], dtype=float)
        x_validation_text = np.asarray(
            [by_id[str(row["request_id"])] for row in validation_rows], dtype=float
        )
        encoder = OneHotEncoder(handle_unknown="ignore", min_frequency=2, sparse_output=False)
        metadata_train = encoder.fit_transform(build_structured_features(train_rows))
        metadata_validation = encoder.transform(build_structured_features(validation_rows))
        payload = {
            "labels": labels,
            "train_rows": train_rows,
            "validation_rows": validation_rows,
            "y_train": [str(row["category"]) for row in train_rows],
            "y_validation": [str(row["category"]) for row in validation_rows],
            "x_train_text": x_train_text,
            "x_validation_text": x_validation_text,
            "metadata_train": metadata_train,
            "metadata_validation": metadata_validation,
        }
        self.memory[key] = payload
        return payload


def _evaluate_fold(store: FoldStore, view: str, repeat: int, fold_index: int, recipe: Recipe) -> dict[str, Any]:
    fold = store.get(view, repeat, fold_index, recipe.instruction)
    labels = fold["labels"]
    meta_train = np.asarray(fold["metadata_train"], dtype=float) * float(recipe.metadata_scale)
    meta_validation = np.asarray(fold["metadata_validation"], dtype=float) * float(recipe.metadata_scale)
    x_train = np.concatenate([fold["x_train_text"], meta_train], axis=1)
    x_validation = np.concatenate([fold["x_validation_text"], meta_validation], axis=1)
    supervised = _fit_calibrated(
        x_train,
        fold["y_train"],
        x_validation,
        labels,
        c=recipe.c,
        seed=SEED + repeat * 10 + fold_index,
    )
    svc_weight, proto_weight, knn_weight = BLENDS[recipe.blend]
    probabilities = svc_weight * supervised
    if proto_weight:
        probabilities += proto_weight * _prototype_probabilities(
            fold["x_train_text"],
            fold["y_train"],
            fold["x_validation_text"],
            labels,
            temperature=recipe.prototype_temperature,
        )
    if knn_weight:
        probabilities += knn_weight * _knn_probabilities(
            fold["x_train_text"],
            fold["y_train"],
            fold["x_validation_text"],
            labels,
            neighbors=recipe.knn_neighbors,
        )
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    metrics = operator_metrics(fold["y_validation"], probabilities, labels)
    return {
        "repeat": repeat,
        "fold": fold_index,
        "metrics": metrics,
        "probabilities": probabilities,
        "fold_data": fold,
    }


def _evaluate_recipe(
    store: FoldStore,
    view: str,
    recipe: Recipe,
    folds: Iterable[tuple[int, int]],
    *,
    keep_predictions: bool = False,
) -> dict[str, Any]:
    records = []
    payloads = []
    for repeat, fold_index in folds:
        result = _evaluate_fold(store, view, repeat, fold_index, recipe)
        records.append(result["metrics"])
        if keep_predictions:
            payloads.append(result)
    return {
        "recipe": recipe.as_dict(),
        "view": view,
        "folds": len(records),
        "metrics": _mean_metrics(records),
        "fold_metrics": records,
        "predictions": payloads if keep_predictions else None,
    }


def _replace(recipe: Recipe, **changes: Any) -> Recipe:
    values = recipe.as_dict()
    values.pop("blend_weights", None)
    values.update(changes)
    return Recipe(**values)


def _unique_recipes(recipes: Sequence[Recipe]) -> list[Recipe]:
    result = []
    seen = set()
    for recipe in recipes:
        key = json.dumps(recipe.as_dict(), sort_keys=True, ensure_ascii=False)
        if key not in seen:
            seen.add(key)
            result.append(recipe)
    return result


def _specialist_fold(
    result: Mapping[str, Any],
    *,
    specialist_c: float = 1.0,
) -> dict[str, Any]:
    fold = result["fold_data"]
    labels = [str(value) for value in fold["labels"]]
    vectorizer = FeatureUnion(
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
    train_x = vectorizer.fit_transform([specialist_text(row) for row in fold["train_rows"]])
    validation_x = vectorizer.transform([specialist_text(row) for row in fold["validation_rows"]])
    model = LinearSVC(
        C=float(specialist_c),
        class_weight="balanced",
        random_state=SEED + int(result["repeat"]) * 10 + int(result["fold"]),
        max_iter=10000,
    )
    model.fit(train_x, fold["y_train"])
    raw = np.asarray(model.decision_function(validation_x), dtype=float)
    classes = [str(value) for value in model.classes_]
    class_index = {label: i for i, label in enumerate(classes)}
    decision = np.full((len(fold["validation_rows"]), len(labels)), -1e9, dtype=float)
    label_index = {label: i for i, label in enumerate(labels)}
    for label, source in class_index.items():
        target = label_index.get(label)
        if target is not None:
            decision[:, target] = raw[:, source]
    return {
        "repeat": result["repeat"],
        "fold": result["fold"],
        "labels": labels,
        "truth": list(fold["y_validation"]),
        "probabilities": np.asarray(result["probabilities"], dtype=float),
        "decision": decision,
    }


def _discover_pairs(specialist_folds: Sequence[Mapping[str, Any]]) -> list[tuple[str, str]]:
    stats: dict[tuple[str, str], dict[str, Any]] = defaultdict(
        lambda: {"corrected": 0, "introduced": 0, "opportunities": 0, "correction_folds": set()}
    )
    for payload in specialist_folds:
        labels = payload["labels"]
        probs = payload["probabilities"]
        decision = payload["decision"]
        truth = payload["truth"]
        order = np.argsort(-probs, axis=1)
        for row_index, row_order in enumerate(order):
            first_i, second_i = int(row_order[0]), int(row_order[1])
            first, second = labels[first_i], labels[second_i]
            pair = tuple(sorted((first, second)))
            margin = float(decision[row_index, second_i] - decision[row_index, first_i])
            if not math.isfinite(margin) or margin <= 0:
                continue
            item = stats[pair]
            item["opportunities"] += 1
            if truth[row_index] == second:
                item["corrected"] += 1
                item["correction_folds"].add((payload["repeat"], payload["fold"]))
            elif truth[row_index] == first:
                item["introduced"] += 1
    selected = []
    for pair, item in sorted(stats.items()):
        corrected = int(item["corrected"])
        introduced = int(item["introduced"])
        if corrected >= 3 and corrected - introduced >= 2 and len(item["correction_folds"]) >= 2:
            selected.append(pair)
    return selected


def _apply_specialist(
    payload: Mapping[str, Any],
    pairs: set[frozenset[str]],
    threshold: float,
) -> tuple[np.ndarray, int, int]:
    labels = payload["labels"]
    probs = np.asarray(payload["probabilities"], dtype=float).copy()
    decision = payload["decision"]
    truth = payload["truth"]
    order = np.argsort(-probs, axis=1)
    corrected = 0
    introduced = 0
    for row_index, row_order in enumerate(order):
        first_i, second_i = int(row_order[0]), int(row_order[1])
        first, second = labels[first_i], labels[second_i]
        if frozenset((first, second)) not in pairs:
            continue
        margin = float(decision[row_index, second_i] - decision[row_index, first_i])
        if not math.isfinite(margin) or margin <= float(threshold):
            continue
        before = first
        probs[row_index, first_i], probs[row_index, second_i] = (
            probs[row_index, second_i],
            probs[row_index, first_i],
        )
        after = second
        if before != truth[row_index] and after == truth[row_index]:
            corrected += 1
        elif before == truth[row_index] and after != truth[row_index]:
            introduced += 1
    return probs, corrected, introduced


def _retune_specialist(base_eval: Mapping[str, Any]) -> dict[str, Any]:
    predictions = base_eval["predictions"] or []
    specialist_folds = [_specialist_fold(item) for item in predictions]
    selected_pairs = _discover_pairs(specialist_folds)
    if not selected_pairs:
        return {
            "enabled": False,
            "pairs": [],
            "threshold": None,
            "metrics": dict(base_eval["metrics"]),
            "corrected_errors": 0,
            "introduced_errors": 0,
            "net_top1_delta": 0.0,
            "macro_f1_delta": 0.0,
            "top3_delta": 0.0,
        }
    pair_set = {frozenset(pair) for pair in selected_pairs}
    candidates = []
    for threshold in (0.0, 0.05, 0.10, 0.15, 0.20, 0.30):
        fold_metrics = []
        corrected = introduced = 0
        for payload in specialist_folds:
            probabilities, c, i = _apply_specialist(payload, pair_set, threshold)
            fold_metrics.append(operator_metrics(payload["truth"], probabilities, payload["labels"]))
            corrected += c
            introduced += i
        metrics = _mean_metrics(fold_metrics)
        candidates.append(
            {
                "threshold": threshold,
                "metrics": metrics,
                "corrected_errors": corrected,
                "introduced_errors": introduced,
            }
        )
    candidates.sort(key=lambda item: _quality_key(item["metrics"]), reverse=True)
    best = candidates[0]
    base = base_eval["metrics"]
    improves = _quality_key(best["metrics"]) > _quality_key(base)
    if not improves:
        return {
            "enabled": False,
            "pairs": selected_pairs,
            "threshold": None,
            "metrics": dict(base),
            "corrected_errors": 0,
            "introduced_errors": 0,
            "net_top1_delta": 0.0,
            "macro_f1_delta": 0.0,
            "top3_delta": 0.0,
            "screen": candidates,
        }
    return {
        "enabled": True,
        "pairs": selected_pairs,
        "threshold": best["threshold"],
        "metrics": best["metrics"],
        "corrected_errors": best["corrected_errors"],
        "introduced_errors": best["introduced_errors"],
        "net_top1_delta": round(best["metrics"]["top1_accuracy"] - base["top1_accuracy"], 9),
        "macro_f1_delta": round(best["metrics"]["macro_f1"] - base["macro_f1"], 9),
        "top3_delta": round(best["metrics"]["top3_accuracy"] - base["top3_accuracy"], 9),
        "screen": candidates,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compact Qwen3-Embedding-4B Lite V5.2 quality sprint")
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=ROOT / "artifacts/gpu_research_v5/embedding_cache",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/qwen4b_lite/tuning.json",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    started = time.time()
    rows = load_v5_rows(DEFAULT_DATASET)
    protocol = load_json(DEFAULT_PROTOCOL)
    contract = load_json(DEFAULT_CONTRACT)
    store = FoldStore(
        rows=rows,
        protocol=protocol,
        contract=contract,
        cache_root=args.cache_root,
        batch_size=args.batch_size,
    )
    screen_folds = [(0, fold) for fold in range(4)]
    full_folds = [(repeat, fold) for repeat in range(3) for fold in range(4)]

    # Exact old matched-reference reproduction: en_concise + pure calibrated LinearSVC C=1.
    matched_reference = Recipe(instruction=BASELINE_INSTRUCTION, c=1.0, blend="svc100")
    matched_eval = _evaluate_recipe(store, "top15", matched_reference, full_folds)
    matched_full43 = _evaluate_recipe(store, "full43", matched_reference, full_folds)

    en_start = Recipe(instruction=BASELINE_INSTRUCTION)
    alt_start = Recipe(instruction=ALT_INSTRUCTION)
    en_screen = _evaluate_recipe(store, "top15", en_start, screen_folds)
    alt_screen = _evaluate_recipe(store, "top15", alt_start, screen_folds)
    current = alt_start if _quality_key(alt_screen["metrics"]) > _quality_key(en_screen["metrics"]) else en_start

    ledger: list[dict[str, Any]] = [en_screen, alt_screen]

    c_candidates = [_replace(current, c=value) for value in (1.0, 1.5, 2.0, 3.0, 4.0, 5.0)]
    c_results = [_evaluate_recipe(store, "top15", recipe, screen_folds) for recipe in c_candidates]
    ledger.extend(c_results)
    c_results.sort(key=lambda item: _quality_key(item["metrics"]), reverse=True)
    current = Recipe(**{k: v for k, v in c_results[0]["recipe"].items() if k != "blend_weights"})

    blend_candidates = [_replace(current, blend=name) for name in BLENDS]
    blend_results = [_evaluate_recipe(store, "top15", recipe, screen_folds) for recipe in blend_candidates]
    ledger.extend(blend_results)
    blend_results.sort(key=lambda item: _quality_key(item["metrics"]), reverse=True)
    current = Recipe(**{k: v for k, v in blend_results[0]["recipe"].items() if k != "blend_weights"})

    metadata_candidates = [_replace(current, metadata_scale=value) for value in (0.75, 1.0, 1.25)]
    metadata_results = [_evaluate_recipe(store, "top15", recipe, screen_folds) for recipe in metadata_candidates]
    ledger.extend(metadata_results)
    metadata_results.sort(key=lambda item: _quality_key(item["metrics"]), reverse=True)
    current = Recipe(**{k: v for k, v in metadata_results[0]["recipe"].items() if k != "blend_weights"})

    neighborhood = [
        current,
        _replace(current, knn_neighbors=7),
        _replace(current, knn_neighbors=11),
        _replace(current, prototype_temperature=0.06),
        _replace(current, prototype_temperature=0.10),
    ]
    neighborhood_results = [_evaluate_recipe(store, "top15", recipe, screen_folds) for recipe in _unique_recipes(neighborhood)]
    ledger.extend(neighborhood_results)

    by_recipe: dict[str, dict[str, Any]] = {}
    for item in ledger:
        key = json.dumps(item["recipe"], ensure_ascii=False, sort_keys=True)
        existing = by_recipe.get(key)
        if existing is None or _quality_key(item["metrics"]) > _quality_key(existing["metrics"]):
            by_recipe[key] = item
    screened = sorted(by_recipe.values(), key=lambda item: _quality_key(item["metrics"]), reverse=True)
    finalists = []
    for item in screened[:3]:
        recipe = Recipe(**{k: v for k, v in item["recipe"].items() if k != "blend_weights"})
        top15 = _evaluate_recipe(store, "top15", recipe, full_folds, keep_predictions=True)
        full43 = _evaluate_recipe(store, "full43", recipe, full_folds, keep_predictions=True)
        finalists.append({"recipe": recipe.as_dict(), "top15": top15, "full43": full43})
    finalists.sort(
        key=lambda item: (
            *_quality_key(item["top15"]["metrics"]),
            float(item["full43"]["metrics"]["macro_f1"]),
        ),
        reverse=True,
    )
    winner = finalists[0]
    specialist_top15 = _retune_specialist(winner["top15"])
    specialist_full43 = _retune_specialist(winner["full43"])

    def clean_eval(value: Mapping[str, Any]) -> dict[str, Any]:
        return {key: item for key, item in value.items() if key != "predictions"}

    payload = {
        "status": "complete",
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "feature_mode": FEATURE_MODE,
        "max_length": MAX_LENGTH,
        "embedding_dim": EMBEDDING_DIM,
        "lockbox_accessed": False,
        "synthetic_rows": 0,
        "elapsed_seconds": round(time.time() - started, 3),
        "embedding_cache_hits": store.embedder.hits,
        "embedding_cache_misses": store.embedder.misses,
        "matched_reference_reproduction": {
            "top15": clean_eval(matched_eval),
            "full43": clean_eval(matched_full43),
        },
        "screen": [clean_eval(item) for item in screened],
        "finalists": [
            {
                "recipe": item["recipe"],
                "top15": clean_eval(item["top15"]),
                "full43": clean_eval(item["full43"]),
            }
            for item in finalists
        ],
        "winner": {
            "recipe": winner["recipe"],
            "top15_base": clean_eval(winner["top15"]),
            "full43_base": clean_eval(winner["full43"]),
            "top15_specialist": specialist_top15,
            "full43_specialist": specialist_full43,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {
        "output": str(args.output),
        "elapsed_seconds": payload["elapsed_seconds"],
        "cache_hits": payload["embedding_cache_hits"],
        "cache_misses": payload["embedding_cache_misses"],
        "matched_top15": matched_eval["metrics"],
        "matched_full43": matched_full43["metrics"],
        "winner": winner["recipe"],
        "winner_top15_base": winner["top15"]["metrics"],
        "winner_full43_base": winner["full43"]["metrics"],
        "winner_top15_specialist": specialist_top15,
        "winner_full43_specialist": specialist_full43,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
