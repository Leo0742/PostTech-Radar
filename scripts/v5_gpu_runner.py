from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import traceback
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import LinearSVC

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.v5_dataset import DEFAULT_DATASET, load_v5_rows  # noqa: E402
from scripts.v5_experiment import (  # noqa: E402
    DEFAULT_CONTRACT,
    DEFAULT_PROTOCOL,
    EXTENDED_INSTRUCTION_REGISTRY,
    build_stage_candidates,
    build_text,
    candidate_quality_key,
    deterministic_run_id,
    labels_for_view,
    load_json,
    operator_metrics,
    truncate_embeddings,
)
from scripts.v5_protocol import group_hash  # noqa: E402

CANDIDATE_IDENTITY_EXCLUDE = {"stage", "repeat", "fold", "seed", "run_id"}
QUALITY_METRICS = ("top1_accuracy", "macro_f1", "top3_accuracy", "true_label_mrr")
STRUCTURED_FIELDS = (
    "user",
    "service",
    "component",
    "request_type",
    "criticality",
    "urgency",
    "priority",
    "service_class",
    "timezone",
    "registration_month",
    "registration_weekday",
    "registration_hour",
)
MISSING = "__MISSING__"


def _sha256_payload(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


class DiskEmbeddingCache:
    def __init__(
        self,
        root: Path,
        backend: Callable[[Sequence[str], Mapping[str, Any]], np.ndarray],
    ) -> None:
        self.root = Path(root)
        self.backend = backend
        self.hits = 0
        self.misses = 0

    @staticmethod
    def _key(texts: Sequence[str], candidate: Mapping[str, Any]) -> str:
        representation = {
            "model_id": candidate.get("model_id"),
            "model_revision": candidate.get("model_revision"),
            "feature_mode": candidate.get("feature_mode"),
            "instruction": candidate.get("instruction"),
            "max_length": int(candidate.get("max_length", 0)),
            "texts": list(texts),
        }
        return _sha256_payload(representation)

    def __call__(self, texts: Sequence[str], candidate: Mapping[str, Any]) -> np.ndarray:
        key = self._key(texts, candidate)
        array_path = self.root / f"{key}.npy"
        meta_path = self.root / f"{key}.json"
        if array_path.exists() and meta_path.exists():
            self.hits += 1
            return np.load(array_path, allow_pickle=False)

        self.misses += 1
        matrix = np.asarray(self.backend(texts, candidate), dtype=np.float32)
        if matrix.ndim != 2 or matrix.shape[0] != len(texts):
            raise ValueError("Embedding backend must return a 2D matrix with one row per text")
        self.root.mkdir(parents=True, exist_ok=True)
        temporary = array_path.with_suffix(".npy.tmp")
        with temporary.open("wb") as stream:
            np.save(stream, matrix, allow_pickle=False)
        temporary.replace(array_path)
        _atomic_write_json(
            meta_path,
            {
                "cache_key": key,
                "rows": int(matrix.shape[0]),
                "width": int(matrix.shape[1]),
                "dtype": str(matrix.dtype),
                "model_id": candidate.get("model_id"),
                "model_revision": candidate.get("model_revision"),
                "feature_mode": candidate.get("feature_mode"),
                "instruction": candidate.get("instruction"),
                "max_length": int(candidate.get("max_length", 0)),
                "text_sha256": _sha256_payload(list(texts)),
            },
        )
        return matrix


def resolve_huggingface_model(model_id: str) -> dict[str, Any]:
    from huggingface_hub import model_info

    info = model_info(model_id)
    card_data = getattr(info, "card_data", None)
    license_value = None
    if card_data is not None:
        license_value = getattr(card_data, "license", None)
        if license_value is None and isinstance(card_data, Mapping):
            license_value = card_data.get("license")
    revision = str(getattr(info, "sha", "") or "")
    if not revision:
        raise RuntimeError(f"Hugging Face did not return an immutable revision for {model_id}")
    return {
        "model_id": model_id,
        "revision": revision,
        "license": license_value,
    }


def pin_candidate_revision(
    candidate: Mapping[str, Any],
    *,
    resolver: Callable[[str], Mapping[str, Any]] = resolve_huggingface_model,
) -> tuple[dict[str, Any], dict[str, Any]]:
    result = dict(candidate)
    if result.get("model_revision"):
        metadata = {
            "model_id": str(result["model_id"]),
            "revision": str(result["model_revision"]),
            "license": None,
            "source": "config_or_promoted_result",
        }
    else:
        metadata = dict(resolver(str(result["model_id"])))
        result["model_revision"] = str(metadata["revision"])
        metadata["source"] = "huggingface_model_info"
    result.pop("run_id", None)
    result["run_id"] = deterministic_run_id(result)
    return result, metadata


class SentenceTransformerBackend:
    def __init__(self, *, batch_size: int = 8) -> None:
        self.requested_batch_size = max(1, int(batch_size))
        self._model: Any = None
        self._model_key: tuple[str, str] | None = None
        self.last_stats: dict[str, Any] = {}
        self.call_stats: list[dict[str, Any]] = []

    def _load_model(self, candidate: Mapping[str, Any]) -> Any:
        import torch
        from sentence_transformers import SentenceTransformer

        model_id = str(candidate["model_id"])
        revision = str(candidate["model_revision"])
        key = (model_id, revision)
        if self._model is not None and self._model_key == key:
            return self._model
        if self._model is not None:
            del self._model
            self._model = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA GPU is required for the V5 GPU runner")
        self._model = SentenceTransformer(
            model_id,
            revision=revision,
            trust_remote_code=True,
            device="cuda",
            model_kwargs={"torch_dtype": torch.float16},
        )
        self._model_key = key
        return self._model

    def __call__(self, texts: Sequence[str], candidate: Mapping[str, Any]) -> np.ndarray:
        import torch

        model = self._load_model(candidate)
        model.max_seq_length = int(candidate["max_length"])
        batch_size = self.requested_batch_size
        started = time.perf_counter()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        attempts: list[int] = []
        while True:
            attempts.append(batch_size)
            try:
                values = model.encode(
                    list(texts),
                    batch_size=batch_size,
                    show_progress_bar=True,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                )
                break
            except RuntimeError as exc:
                if "out of memory" not in str(exc).lower() or batch_size <= 1:
                    raise
                batch_size = max(1, batch_size // 2)
                torch.cuda.empty_cache()
        elapsed = time.perf_counter() - started
        peak = int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0
        self.last_stats = {
            "rows": len(texts),
            "seconds": round(elapsed, 6),
            "requested_batch_size": self.requested_batch_size,
            "used_batch_size": batch_size,
            "batch_attempts": attempts,
            "peak_vram_bytes": peak,
        }
        self.call_stats.append(dict(self.last_stats))
        return np.asarray(values, dtype=np.float32)


def output_dir_for_run(results_root: Path, candidate: Mapping[str, Any]) -> Path:
    return Path(results_root) / str(candidate["stage"]) / str(candidate["run_id"])


def should_skip_run(output_dir: Path) -> bool:
    path = Path(output_dir) / "result.json"
    if not path.exists():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return payload.get("status") == "complete"


def load_completed_stage_results(results_root: Path, stage: str) -> list[dict[str, Any]]:
    stage_root = Path(results_root) / stage
    if not stage_root.exists():
        return []
    records: list[dict[str, Any]] = []
    for path in sorted(stage_root.glob("*/result.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if payload.get("status") == "complete" and isinstance(payload.get("candidate"), dict):
            records.append(payload)
    return records


def _candidate_identity(candidate: Mapping[str, Any]) -> str:
    stable = {key: value for key, value in candidate.items() if key not in CANDIDATE_IDENTITY_EXCLUDE}
    return json.dumps(stable, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def select_promoted_candidates(results: Sequence[Mapping[str, Any]], *, top_n: int) -> list[dict[str, Any]]:
    if top_n < 1:
        raise ValueError("top_n must be >= 1")
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for result in results:
        if result.get("status") != "complete" or not isinstance(result.get("candidate"), Mapping):
            continue
        metrics = result.get("metrics")
        if not isinstance(metrics, Mapping):
            continue
        grouped[_candidate_identity(result["candidate"])].append(result)

    aggregated: list[dict[str, Any]] = []
    for records in grouped.values():
        metric_values = {
            name: round(mean(float(record["metrics"].get(name, 0.0)) for record in records), 12)
            for name in QUALITY_METRICS
        }
        candidate = dict(records[0]["candidate"])
        aggregated.append(
            {
                "candidate": candidate,
                "aggregate_metrics": metric_values,
                "fold_results": len(records),
            }
        )
    aggregated.sort(key=lambda item: candidate_quality_key(item["aggregate_metrics"]), reverse=True)
    return aggregated[:top_n]


def _stage_config(config: Mapping[str, Any], stage_name: str) -> Mapping[str, Any]:
    for stage in config.get("stages", []):
        if stage.get("name") == stage_name:
            return stage
    raise ValueError(f"Unknown stage: {stage_name}")


def resolve_stage_candidates(
    config: Mapping[str, Any],
    stage_name: str,
    *,
    protocol: Mapping[str, Any],
    results_root: Path,
) -> list[dict[str, Any]]:
    stage = _stage_config(config, stage_name)
    raw = build_stage_candidates(config, stage_name, protocol=protocol)
    source_stage = stage.get("inherit_best_from") or stage.get("inherit_top_n_from")
    if not source_stage:
        return raw

    top_n = int(stage.get("top_n", 1)) if stage.get("inherit_top_n_from") else 1
    previous = load_completed_stage_results(results_root, str(source_stage))
    promoted = select_promoted_candidates(previous, top_n=top_n)
    if not promoted:
        raise ValueError(f"Stage {stage_name} requires completed results from {source_stage}")

    inherit_fields = [str(value) for value in stage.get("inherit_fields", [])]
    resolved: dict[str, dict[str, Any]] = {}
    for promoted_item in promoted:
        source_candidate = promoted_item["candidate"]
        for base in raw:
            candidate = dict(base)
            for field in inherit_fields:
                if field in source_candidate:
                    candidate[field] = source_candidate[field]
            candidate.pop("run_id", None)
            candidate["run_id"] = deterministic_run_id(candidate)
            resolved[candidate["run_id"]] = candidate
    return sorted(resolved.values(), key=lambda item: item["run_id"])


def _clean_structured(value: Any) -> str:
    if value is None:
        return MISSING
    text = " ".join(str(value).split())
    return text or MISSING


def build_structured_features(rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
    result: list[list[str]] = []
    for row in rows:
        registration = datetime.fromisoformat(str(row["registration_date"]).replace("Z", "+00:00"))
        values = [
            _clean_structured(row.get("user")),
            _clean_structured(row.get("service")),
            _clean_structured(row.get("component")),
            _clean_structured(row.get("request_type")),
            _clean_structured(row.get("criticality")),
            _clean_structured(row.get("urgency")),
            _clean_structured(row.get("priority")),
            _clean_structured(row.get("service_class")),
            _clean_structured(row.get("timezone")),
            str(registration.month),
            str(registration.weekday()),
            str(registration.hour),
        ]
        result.append(values)
    return np.asarray(result, dtype=object)


def _apply_instruction(texts: Sequence[str], instruction_key: str) -> list[str]:
    instruction = EXTENDED_INSTRUCTION_REGISTRY[instruction_key]
    if not instruction:
        return [str(text) for text in texts]
    return [f"Instruct: {instruction}\nQuery: {text}" for text in texts]


def _normalize_rows(values: np.ndarray) -> np.ndarray:
    matrix = np.asarray(values, dtype=float)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return np.divide(matrix, norms, out=np.zeros_like(matrix), where=norms > 0)


def _softmax(values: np.ndarray, *, temperature: float = 1.0) -> np.ndarray:
    temperature = max(float(temperature), 1e-6)
    scaled = np.asarray(values, dtype=float) / temperature
    scaled -= np.max(scaled, axis=1, keepdims=True)
    exp = np.exp(scaled)
    return exp / exp.sum(axis=1, keepdims=True)


def _align_probabilities(model: Any, values: np.ndarray, labels: Sequence[str]) -> np.ndarray:
    local = np.asarray(model.predict_proba(values), dtype=float)
    classes = [str(value) for value in model.classes_]
    index = {str(label): position for position, label in enumerate(labels)}
    result = np.zeros((len(values), len(labels)), dtype=float)
    for source, label in enumerate(classes):
        if label in index:
            result[:, index[label]] = local[:, source]
    sums = result.sum(axis=1, keepdims=True)
    return np.divide(result, sums, out=np.zeros_like(result), where=sums > 0)


def _base_logreg(seed: int) -> LogisticRegression:
    return LogisticRegression(
        max_iter=3000,
        class_weight="balanced",
        C=1.0,
        random_state=seed,
    )


def _fit_predict_supervised_head(
    head: str,
    x_train: np.ndarray,
    y_train: Sequence[str],
    x_validation: np.ndarray,
    labels: Sequence[str],
    *,
    seed: int,
) -> tuple[np.ndarray, str]:
    if head in {"logreg", "metadata_fusion"}:
        model = _base_logreg(seed)
        model.fit(x_train, y_train)
        return _align_probabilities(model, x_validation, labels), "logreg"

    linearsvc_variants = {
        "calibrated_linearsvc": (1.0, "balanced", "sigmoid"),
        "calibrated_linearsvc_c0_5": (0.5, "balanced", "sigmoid"),
        "calibrated_linearsvc_c2": (2.0, "balanced", "sigmoid"),
        "calibrated_linearsvc_c4": (4.0, "balanced", "sigmoid"),
        "calibrated_linearsvc_unweighted": (1.0, None, "sigmoid"),
        "calibrated_linearsvc_isotonic": (1.0, "balanced", "isotonic"),
    }
    if head == "calibrated_logreg" or head in linearsvc_variants:
        min_count = min(Counter(str(value) for value in y_train).values())
        if min_count < 2:
            model = _base_logreg(seed)
            model.fit(x_train, y_train)
            return _align_probabilities(model, x_validation, labels), "logreg_fallback_insufficient_calibration_rows"
        cv = min(3, min_count)
        if head == "calibrated_logreg":
            estimator: Any = _base_logreg(seed)
            method = "sigmoid"
        else:
            c_value, class_weight, method = linearsvc_variants[head]
            estimator = LinearSVC(
                C=c_value,
                class_weight=class_weight,
                random_state=seed,
                max_iter=10000,
            )
        model = CalibratedClassifierCV(estimator=estimator, method=method, cv=cv)
        model.fit(x_train, y_train)
        return _align_probabilities(model, x_validation, labels), head

    if head == "shallow_mlp":
        label_to_index = {str(label): index for index, label in enumerate(labels)}
        y_train_encoded = np.asarray([label_to_index[str(value)] for value in y_train], dtype=int)
        model = MLPClassifier(
            hidden_layer_sizes=(128,),
            activation="relu",
            alpha=1e-4,
            max_iter=350,
            early_stopping=True,
            random_state=seed,
        )
        model.fit(x_train, y_train_encoded)
        local = np.asarray(model.predict_proba(x_validation), dtype=float)
        result = np.zeros((len(x_validation), len(labels)), dtype=float)
        for source, class_index in enumerate(model.classes_):
            result[:, int(class_index)] = local[:, source]
        sums = result.sum(axis=1, keepdims=True)
        result = np.divide(result, sums, out=np.zeros_like(result), where=sums > 0)
        return result, head

    if head == "knn":
        model = KNeighborsClassifier(
            n_neighbors=max(1, min(9, len(y_train))),
            weights="distance",
            metric="cosine",
            algorithm="brute",
        )
        model.fit(x_train, y_train)
        return _align_probabilities(model, x_validation, labels), head

    if head == "centroid":
        train = _normalize_rows(x_train)
        validation = _normalize_rows(x_validation)
        centroids = []
        for label in labels:
            indices = [index for index, value in enumerate(y_train) if str(value) == str(label)]
            if not indices:
                centroids.append(np.zeros(train.shape[1], dtype=float))
            else:
                centroids.append(_normalize_rows(np.mean(train[indices], axis=0, keepdims=True))[0])
        scores = validation @ np.asarray(centroids, dtype=float).T
        return _softmax(scores, temperature=0.08), head

    raise ValueError(f"Unsupported supervised head: {head}")


def _metadata_probabilities(
    train_rows: Sequence[Mapping[str, Any]],
    validation_rows: Sequence[Mapping[str, Any]],
    y_train: Sequence[str],
    labels: Sequence[str],
    *,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, OneHotEncoder]:
    encoder = OneHotEncoder(handle_unknown="ignore", min_frequency=2, sparse_output=False)
    train_encoded = encoder.fit_transform(build_structured_features(train_rows))
    validation_encoded = encoder.transform(build_structured_features(validation_rows))
    model = _base_logreg(seed)
    model.fit(train_encoded, y_train)
    return _align_probabilities(model, validation_encoded, labels), train_encoded, encoder


def _catboost_metadata_probabilities(
    train_rows: Sequence[Mapping[str, Any]],
    validation_rows: Sequence[Mapping[str, Any]],
    y_train: Sequence[str],
    labels: Sequence[str],
    *,
    seed: int,
) -> np.ndarray:
    from catboost import CatBoostClassifier

    train_values = build_structured_features(train_rows).astype(str)
    validation_values = build_structured_features(validation_rows).astype(str)
    model = CatBoostClassifier(
        loss_function="MultiClass",
        iterations=250,
        depth=6,
        learning_rate=0.06,
        random_seed=seed,
        verbose=False,
        allow_writing_files=False,
        thread_count=-1,
    )
    model.fit(train_values, list(y_train), cat_features=list(range(train_values.shape[1])))
    local = np.asarray(model.predict_proba(validation_values), dtype=float)
    classes = [str(value) for value in model.classes_]
    index = {str(label): position for position, label in enumerate(labels)}
    result = np.zeros((len(validation_rows), len(labels)), dtype=float)
    for source, label in enumerate(classes):
        if label in index:
            result[:, index[label]] = local[:, source]
    sums = result.sum(axis=1, keepdims=True)
    return np.divide(result, sums, out=np.zeros_like(result), where=sums > 0)


def _prototype_probabilities(
    x_train: np.ndarray,
    y_train: Sequence[str],
    x_validation: np.ndarray,
    labels: Sequence[str],
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
    return _softmax(validation @ np.asarray(centroids, dtype=float).T, temperature=0.08)


def _knn_probabilities(
    x_train: np.ndarray,
    y_train: Sequence[str],
    x_validation: np.ndarray,
    labels: Sequence[str],
) -> np.ndarray:
    model = KNeighborsClassifier(
        n_neighbors=max(1, min(9, len(y_train))),
        weights="distance",
        metric="cosine",
        algorithm="brute",
    )
    model.fit(x_train, y_train)
    return _align_probabilities(model, x_validation, labels)


def _prediction_payload(
    validation_rows: Sequence[Mapping[str, Any]],
    truth: Sequence[str],
    probabilities: np.ndarray,
    labels: Sequence[str],
) -> dict[str, Any]:
    matrix = np.asarray(probabilities, dtype=float)
    if matrix.ndim != 2 or matrix.shape != (len(validation_rows), len(labels)):
        raise ValueError("Prediction matrix shape does not match validation rows and labels")
    top_k = min(3, len(labels))
    order = np.argsort(-matrix, axis=1)[:, :top_k]
    records: list[dict[str, Any]] = []
    for row_index, row in enumerate(validation_rows):
        ranked = [
            {
                "label": str(labels[int(label_index)]),
                "probability": round(float(matrix[row_index, int(label_index)]), 8),
            }
            for label_index in order[row_index]
        ]
        records.append(
            {
                "request_id": str(row["request_id"]),
                "truth": str(truth[row_index]),
                "top1": ranked[0]["label"],
                "top1_confidence": ranked[0]["probability"],
                "top3": ranked,
                "probabilities": [round(float(value), 8) for value in matrix[row_index]],
            }
        )
    return {
        "prediction_labels": [str(label) for label in labels],
        "predictions": records,
    }


def _text_metadata_features(
    train_text: np.ndarray,
    validation_text: np.ndarray,
    train_rows: Sequence[Mapping[str, Any]],
    validation_rows: Sequence[Mapping[str, Any]],
) -> tuple[np.ndarray, np.ndarray]:
    encoder = OneHotEncoder(handle_unknown="ignore", min_frequency=2, sparse_output=False)
    train_metadata = encoder.fit_transform(build_structured_features(train_rows))
    validation_metadata = encoder.transform(build_structured_features(validation_rows))
    return (
        np.concatenate([train_text, train_metadata], axis=1),
        np.concatenate([validation_text, validation_metadata], axis=1),
    )


def _oof_stack_probabilities(
    x_train_text: np.ndarray,
    x_validation_text: np.ndarray,
    train_rows: Sequence[Mapping[str, Any]],
    validation_rows: Sequence[Mapping[str, Any]],
    y_train: Sequence[str],
    labels: Sequence[str],
    *,
    seed: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    groups = [
        group_hash(row.get("description"), request_id=str(row["request_id"]))
        for row in train_rows
    ]
    groups_by_label: dict[str, set[str]] = defaultdict(set)
    for label, group in zip(y_train, groups, strict=True):
        groups_by_label[str(label)].add(str(group))
    min_groups = min((len(value) for value in groups_by_label.values()), default=0)
    n_splits = min(4, min_groups)
    if n_splits < 2:
        base_train, base_validation = _text_metadata_features(
            x_train_text, x_validation_text, train_rows, validation_rows
        )
        probabilities, effective = _fit_predict_supervised_head(
            "calibrated_linearsvc",
            base_train,
            y_train,
            base_validation,
            labels,
            seed=seed,
        )
        return probabilities, {
            "effective_head": f"oof_stack_fallback:{effective}",
            "inner_splits": n_splits,
            "reason": "insufficient independent groups for leakage-safe inner OOF",
        }

    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    y_array = np.asarray([str(value) for value in y_train], dtype=object)
    groups_array = np.asarray(groups, dtype=object)
    branch_count = 5
    oof = np.zeros((len(train_rows), len(labels) * branch_count), dtype=float)
    coverage = np.zeros(len(train_rows), dtype=bool)

    for inner_fold, (inner_train, inner_validation) in enumerate(
        splitter.split(x_train_text, y_array, groups_array)
    ):
        inner_train_rows = [train_rows[int(index)] for index in inner_train]
        inner_validation_rows = [train_rows[int(index)] for index in inner_validation]
        inner_y = [str(y_train[int(index)]) for index in inner_train]
        inner_x_train_text = x_train_text[inner_train]
        inner_x_validation_text = x_train_text[inner_validation]
        inner_x_train, inner_x_validation = _text_metadata_features(
            inner_x_train_text,
            inner_x_validation_text,
            inner_train_rows,
            inner_validation_rows,
        )
        embed_meta, _ = _fit_predict_supervised_head(
            "calibrated_linearsvc",
            inner_x_train,
            inner_y,
            inner_x_validation,
            labels,
            seed=seed + inner_fold,
        )
        metadata, _metadata_train, _encoder = _metadata_probabilities(
            inner_train_rows,
            inner_validation_rows,
            inner_y,
            labels,
            seed=seed + inner_fold,
        )
        catboost = _catboost_metadata_probabilities(
            inner_train_rows,
            inner_validation_rows,
            inner_y,
            labels,
            seed=seed + inner_fold,
        )
        prototype = _prototype_probabilities(
            inner_x_train_text,
            inner_y,
            inner_x_validation_text,
            labels,
        )
        knn = _knn_probabilities(
            inner_x_train_text,
            inner_y,
            inner_x_validation_text,
            labels,
        )
        oof[inner_validation] = np.concatenate(
            [embed_meta, metadata, catboost, prototype, knn], axis=1
        )
        coverage[inner_validation] = True

    if not bool(np.all(coverage)):
        raise RuntimeError("Inner OOF stacking did not cover every outer-train row")

    meta = LogisticRegression(
        max_iter=3000,
        class_weight="balanced",
        C=0.5,
        random_state=seed,
    )
    meta.fit(oof, y_array)

    full_train, full_validation = _text_metadata_features(
        x_train_text, x_validation_text, train_rows, validation_rows
    )
    embed_meta_validation, _ = _fit_predict_supervised_head(
        "calibrated_linearsvc",
        full_train,
        y_train,
        full_validation,
        labels,
        seed=seed,
    )
    metadata_validation, _metadata_train, _encoder = _metadata_probabilities(
        train_rows,
        validation_rows,
        y_train,
        labels,
        seed=seed,
    )
    catboost_validation = _catboost_metadata_probabilities(
        train_rows,
        validation_rows,
        y_train,
        labels,
        seed=seed,
    )
    prototype_validation = _prototype_probabilities(
        x_train_text,
        y_train,
        x_validation_text,
        labels,
    )
    knn_validation = _knn_probabilities(
        x_train_text,
        y_train,
        x_validation_text,
        labels,
    )
    validation_meta = np.concatenate(
        [
            embed_meta_validation,
            metadata_validation,
            catboost_validation,
            prototype_validation,
            knn_validation,
        ],
        axis=1,
    )
    probabilities = _align_probabilities(meta, validation_meta, labels)
    return probabilities, {
        "effective_head": "oof_stack_logreg",
        "inner_splits": n_splits,
        "branches": ["embed_plus_onehot", "metadata_logreg", "metadata_catboost", "prototype", "knn"],
        "meta_classifier": {"type": "logreg", "C": 0.5, "class_weight": "balanced"},
        "grouping": "normalized_description_hash",
        "oof_only_meta_training": True,
    }


def evaluate_candidate_with_embedder(
    candidate: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    *,
    train_ids: Sequence[str],
    validation_ids: Sequence[str],
    labels: Sequence[str],
    embedder: Callable[[Sequence[str], Mapping[str, Any]], np.ndarray],
) -> dict[str, Any]:
    label_set = {str(value) for value in labels}
    rows_by_id = {str(row["request_id"]): row for row in rows}
    train_rows = [rows_by_id[str(value)] for value in train_ids if str(value) in rows_by_id and str(rows_by_id[str(value)]["category"]) in label_set]
    validation_rows = [rows_by_id[str(value)] for value in validation_ids if str(value) in rows_by_id and str(rows_by_id[str(value)]["category"]) in label_set]
    if not train_rows or not validation_rows:
        raise ValueError("Candidate fold has no rows after applying the category view")

    feature_mode = str(candidate["feature_mode"])
    instruction = str(candidate["instruction"])
    y_train = [str(row["category"]) for row in train_rows]
    y_validation = [str(row["category"]) for row in validation_rows]
    seed = int(candidate.get("seed", 20260917))

    if feature_mode == "metadata_only_fallback":
        probabilities, _metadata_train, _encoder = _metadata_probabilities(
            train_rows,
            validation_rows,
            y_train,
            labels,
            seed=seed,
        )
        metrics = operator_metrics(y_validation, probabilities, labels)
        result = {
            "status": "complete",
            "candidate": dict(candidate),
            "metrics": metrics,
            "effective_head": "metadata_logreg_fallback",
            "partition": {
                "train_rows": len(train_rows),
                "validation_rows": len(validation_rows),
            },
            "real_only_evaluation": True,
            "synthetic_rows_in_validation": 0,
            "internal_lockbox_accessed": False,
        }
        result.update(_prediction_payload(validation_rows, y_validation, probabilities, labels))
        return result

    all_rows_by_id = {str(row["request_id"]): row for row in [*train_rows, *validation_rows]}
    ordered_rows = [all_rows_by_id[key] for key in sorted(all_rows_by_id)]
    texts = [build_text(row, feature_mode=feature_mode) for row in ordered_rows]
    embedded = np.asarray(embedder(_apply_instruction(texts, instruction), candidate), dtype=float)
    if embedded.shape[0] != len(ordered_rows):
        raise ValueError("Embedder row count does not match requested texts")
    embedded = _normalize_rows(truncate_embeddings(embedded, int(candidate["embedding_dim"])))
    embedding_by_id = {
        str(row["request_id"]): embedded[index]
        for index, row in enumerate(ordered_rows)
    }
    x_train = np.asarray([embedding_by_id[str(row["request_id"])] for row in train_rows], dtype=float)
    x_validation = np.asarray([embedding_by_id[str(row["request_id"])] for row in validation_rows], dtype=float)
    x_train_text = x_train
    x_validation_text = x_validation
    head = str(candidate["head"])

    if feature_mode == "separate_metadata_text":
        encoder = OneHotEncoder(handle_unknown="ignore", min_frequency=2, sparse_output=False)
        metadata_train = encoder.fit_transform(build_structured_features(train_rows))
        metadata_validation = encoder.transform(build_structured_features(validation_rows))
        x_train = np.concatenate([x_train, metadata_train], axis=1)
        x_validation = np.concatenate([x_validation, metadata_validation], axis=1)

    stack_provenance: dict[str, Any] | None = None
    if head == "oof_stack_logreg":
        probabilities, stack_provenance = _oof_stack_probabilities(
            x_train_text,
            x_validation_text,
            train_rows,
            validation_rows,
            y_train,
            labels,
            seed=seed,
        )
        effective_head = str(stack_provenance["effective_head"])
    elif head == "label_similarity":
        label_embeddings = np.asarray(embedder(_apply_instruction([str(label) for label in labels], instruction), candidate), dtype=float)
        label_embeddings = _normalize_rows(truncate_embeddings(label_embeddings, int(candidate["embedding_dim"])))
        probabilities = _softmax(_normalize_rows(x_validation_text) @ label_embeddings.T, temperature=0.06)
        effective_head = head
    else:
        special_base_head = {
            "calibrated_linearsvc_meta20": ("metadata", 0.20),
            "calibrated_linearsvc_meta35": ("metadata", 0.35),
            "calibrated_linearsvc_catboost20": ("catboost", 0.20),
            "calibrated_linearsvc_proto10": ("prototype", 0.10),
            "calibrated_linearsvc_knn10": ("knn", 0.10),
        }
        requested_head = head
        proto_knn_ensemble = requested_head == "calibrated_linearsvc_proto5_knn5"
        supervised_head = "calibrated_linearsvc" if requested_head in special_base_head or proto_knn_ensemble else ("logreg" if head == "metadata_fusion" else head)
        probabilities, effective_head = _fit_predict_supervised_head(
            supervised_head,
            x_train,
            y_train,
            x_validation,
            labels,
            seed=seed,
        )
        if requested_head in special_base_head:
            signal, weight = special_base_head[requested_head]
            if signal == "metadata":
                auxiliary, _metadata_train, _encoder = _metadata_probabilities(
                    train_rows, validation_rows, y_train, labels, seed=seed
                )
            elif signal == "catboost":
                auxiliary = _catboost_metadata_probabilities(
                    train_rows, validation_rows, y_train, labels, seed=seed
                )
            elif signal == "prototype":
                auxiliary = _prototype_probabilities(x_train_text, y_train, x_validation_text, labels)
            else:
                auxiliary = _knn_probabilities(x_train_text, y_train, x_validation_text, labels)
            probabilities = (1.0 - weight) * probabilities + weight * auxiliary
            probabilities /= probabilities.sum(axis=1, keepdims=True)
            effective_head = f"calibrated_linearsvc+{signal}_{weight:.2f}"
        elif proto_knn_ensemble:
            prototype = _prototype_probabilities(x_train_text, y_train, x_validation_text, labels)
            knn = _knn_probabilities(x_train_text, y_train, x_validation_text, labels)
            probabilities = 0.90 * probabilities + 0.05 * prototype + 0.05 * knn
            probabilities /= probabilities.sum(axis=1, keepdims=True)
            effective_head = "calibrated_linearsvc+prototype_0.05+knn_0.05"
        if feature_mode == "metadata_prior_text" or head == "metadata_fusion":
            metadata_probs, _metadata_train, _encoder = _metadata_probabilities(
                train_rows,
                validation_rows,
                y_train,
                labels,
                seed=seed,
            )
            probabilities = 0.75 * probabilities + 0.25 * metadata_probs
            probabilities /= probabilities.sum(axis=1, keepdims=True)
            effective_head = f"{effective_head}+metadata_prior_0.25"

    metrics = operator_metrics(y_validation, probabilities, labels)
    training_support = Counter(str(value) for value in y_train)
    evidence_bands = {}
    for name, predicate in {
        "zero_real_train": lambda value: value == 0,
        "single_real_anchor": lambda value: value == 1,
        "two_to_five_real": lambda value: 2 <= value <= 5,
    }.items():
        band_labels = [str(label) for label in labels if predicate(training_support[str(label)])]
        values = [float(metrics["per_class"][label]["f1"]) for label in band_labels]
        evidence_bands[name] = {
            "labels": band_labels,
            "macro_f1": round(float(mean(values)), 6) if values else None,
            "real_training_support": {label: training_support[label] for label in band_labels},
        }
    result = {
        "status": "complete",
        "candidate": dict(candidate),
        "metrics": metrics,
        "training_evidence_bands": evidence_bands,
        "effective_head": effective_head,
        "partition": {
            "train_rows": len(train_rows),
            "validation_rows": len(validation_rows),
        },
        "real_only_evaluation": True,
        "synthetic_rows_in_validation": 0,
        "internal_lockbox_accessed": False,
    }
    if stack_provenance is not None:
        result["stack_provenance"] = stack_provenance
    result.update(_prediction_payload(validation_rows, y_validation, probabilities, labels))
    return result


def _fold_for_candidate(protocol: Mapping[str, Any], candidate: Mapping[str, Any]) -> Mapping[str, Any]:
    repeat = int(candidate["repeat"])
    fold = int(candidate["fold"])
    for item in protocol["folds"]:
        if int(item["repeat"]) == repeat and int(item["fold"]) == fold:
            return item
    raise ValueError(f"Protocol does not contain repeat={repeat}, fold={fold}")


def execute_stage(
    config: Mapping[str, Any],
    stage_name: str,
    *,
    protocol: Mapping[str, Any],
    contract: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    results_root: Path,
    cache_root: Path,
    resolver: Callable[[str], Mapping[str, Any]] = resolve_huggingface_model,
    backend: Callable[[Sequence[str], Mapping[str, Any]], np.ndarray] | None = None,
    batch_size: int = 8,
    force: bool = False,
) -> dict[str, Any]:
    raw_candidates = resolve_stage_candidates(
        config,
        stage_name,
        protocol=protocol,
        results_root=results_root,
    )
    runtime_backend: Callable[[Sequence[str], Mapping[str, Any]], np.ndarray]
    if backend is None:
        runtime_backend = SentenceTransformerBackend(batch_size=batch_size)
    else:
        runtime_backend = backend
    cached_embedder = DiskEmbeddingCache(cache_root, runtime_backend)

    completed = 0
    skipped = 0
    failed = 0
    run_records: list[dict[str, Any]] = []
    stage_started = time.time()

    for raw_candidate in raw_candidates:
        try:
            candidate, model_metadata = pin_candidate_revision(raw_candidate, resolver=resolver)
        except Exception as exc:
            failed += 1
            run_records.append(
                {
                    "status": "failed_before_run",
                    "candidate": dict(raw_candidate),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            continue

        output_dir = output_dir_for_run(results_root, candidate)
        if should_skip_run(output_dir) and not force:
            skipped += 1
            run_records.append({"status": "skipped", "run_id": candidate["run_id"]})
            continue

        output_dir.mkdir(parents=True, exist_ok=True)
        _atomic_write_json(
            output_dir / "status.json",
            {
                "status": "running",
                "run_id": candidate["run_id"],
                "stage": stage_name,
                "started_at_unix": time.time(),
            },
        )
        started = time.perf_counter()
        backend_calls_before = len(getattr(runtime_backend, "call_stats", []))
        try:
            fold = _fold_for_candidate(protocol, candidate)
            labels = labels_for_view(contract, str(candidate["view"]))
            result = evaluate_candidate_with_embedder(
                candidate,
                rows,
                train_ids=fold["train_request_ids"],
                validation_ids=fold["validation_request_ids"],
                labels=labels,
                embedder=cached_embedder,
            )
            elapsed = time.perf_counter() - started
            result["provenance"] = {
                "protocol_version": protocol.get("protocol_version", "5.1"),
                "dataset_sha256": protocol.get("dataset_sha256"),
                "split_sha256": protocol.get("split_sha256"),
                "source_xlsx_sha256": contract.get("dataset", {}).get("sha256"),
                "model": model_metadata,
                "stage": stage_name,
                "run_id": candidate["run_id"],
            }
            result["runtime"] = {
                "seconds": round(elapsed, 6),
                "embedding_cache_hits_total": cached_embedder.hits,
                "embedding_cache_misses_total": cached_embedder.misses,
                "embedding_backend_last_stats": getattr(runtime_backend, "last_stats", {}),
                "embedding_backend_call_stats": list(getattr(runtime_backend, "call_stats", []))[backend_calls_before:],
            }
            _atomic_write_json(output_dir / "result.json", result)
            _atomic_write_json(
                output_dir / "status.json",
                {
                    "status": "complete",
                    "run_id": candidate["run_id"],
                    "stage": stage_name,
                    "seconds": round(elapsed, 6),
                },
            )
            completed += 1
            run_records.append(
                {
                    "status": "complete",
                    "run_id": candidate["run_id"],
                    "metrics": result["metrics"],
                }
            )
        except Exception as exc:
            elapsed = time.perf_counter() - started
            failure = {
                "status": "failed",
                "candidate": candidate,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
                "seconds": round(elapsed, 6),
                "internal_lockbox_accessed": False,
            }
            _atomic_write_json(output_dir / "result.json", failure)
            _atomic_write_json(
                output_dir / "status.json",
                {
                    "status": "failed",
                    "run_id": candidate["run_id"],
                    "stage": stage_name,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
            )
            failed += 1
            run_records.append(
                {
                    "status": "failed",
                    "run_id": candidate["run_id"],
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

    summary = {
        "server": config.get("server"),
        "stage": stage_name,
        "candidate_count": len(raw_candidates),
        "completed": completed,
        "skipped": skipped,
        "failed": failed,
        "seconds": round(time.time() - stage_started, 6),
        "embedding_cache": {
            "hits": cached_embedder.hits,
            "misses": cached_embedder.misses,
        },
        "runs": run_records,
    }
    _atomic_write_json(Path(results_root) / stage_name / "stage_summary.json", summary)
    return summary


def _results_root(config: Mapping[str, Any]) -> Path:
    root = Path(str(config.get("artifact_root", "artifacts/gpu_research_v5/runs")))
    return root if root.is_absolute() else PROJECT_ROOT / root


def main() -> None:
    parser = argparse.ArgumentParser(description="PostTech Radar V5.1 staged GPU experiment runner")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--stage", required=True)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--results-root", type=Path)
    parser.add_argument("--cache-root", type=Path, default=PROJECT_ROOT / "artifacts" / "gpu_research_v5" / "embedding_cache")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config = load_json(args.config)
    protocol = load_json(args.protocol)
    results_root = args.results_root or _results_root(config)
    candidates = resolve_stage_candidates(
        config,
        args.stage,
        protocol=protocol,
        results_root=results_root,
    )
    if args.dry_run:
        print(
            json.dumps(
                {
                    "server": config.get("server"),
                    "stage": args.stage,
                    "candidate_count": len(candidates),
                    "results_root": str(results_root),
                    "candidates": candidates,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    contract = load_json(args.contract)
    rows = load_v5_rows(args.dataset)
    summary = execute_stage(
        config,
        args.stage,
        protocol=protocol,
        contract=contract,
        rows=rows,
        results_root=results_root,
        cache_root=args.cache_root,
        batch_size=args.batch_size,
        force=args.force,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if summary["completed"] + summary["skipped"] == 0 and summary["failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
