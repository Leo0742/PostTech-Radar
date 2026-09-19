from __future__ import annotations

import hashlib
import itertools
import json
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import balanced_accuracy_score, f1_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROTOCOL = PROJECT_ROOT / "artifacts" / "gpu_research_v5" / "protocol" / "protocol.json"
DEFAULT_CONTRACT = PROJECT_ROOT / "artifacts" / "gpu_research_v5" / "data_contract.json"

EMBEDDING_DIMS = (4096, 3072, 2048, 1024)
SEQUENCE_LENGTHS = (128, 256, 384, 512, 768, 1024)

INSTRUCTION_REGISTRY: dict[str, str] = {
    "none": "",
    "ru_concise": "Представь запрос Service Desk так, чтобы запросы одной категории были близки, а похожие категории различались.",
    "en_concise": "Represent this Service Desk request so tickets from the same category are close and confusable categories are separated.",
    "bilingual": "Представь запрос Service Desk по смыслу категории. Represent the request so same-category tickets are close and confusable categories are separated.",
    "taxonomy_aware": "Encode the Service Desk intent for category classification across the known request taxonomy; separate semantically similar but distinct request categories.",
    "posttech_service_desk": "Represent this Russian PostTech Service Desk request for request-category recommendation to a human operator; keep same-category tickets close and separate commonly confused categories.",
    "posttech_tight_a": "Represent this Russian Service Desk ticket for intent classification. Tickets from the same PostTech category should be close and confusing categories should be separated.",
    "posttech_tight_b": "Encode this support request so that its embedding identifies the correct Service Desk request category, especially distinguishing semantically similar Russian postal IT issues.",
    "posttech_tight_c": "Represent the meaning of this Russian technical support ticket for fine-grained intent classification. Ignore irrelevant wording and emphasize the details that distinguish neighboring request categories.",
    "posttech_ru_noisy": "Представь короткое или шумное русское обращение Service Desk так, чтобы по смыслу точно определить категорию ПочтаТех и отделить её от похожих соседних категорий. Игнорируй лишние формулировки и выделяй различающие детали.",
}

FEATURE_MODES = {
    "text_only",
    "metadata_prefix_text",
    "separate_metadata_text",
    "metadata_prior_text",
    "metadata_only_fallback",
    "text_only_fallback",
}

HEAD_REGISTRY = {
    "logreg",
    "calibrated_logreg",
    "calibrated_linearsvc",
    "shallow_mlp",
    "centroid",
    "knn",
    "label_similarity",
    "metadata_fusion",
    "calibrated_linearsvc_c0_5",
    "calibrated_linearsvc_c2",
    "calibrated_linearsvc_c4",
    "calibrated_linearsvc_unweighted",
    "calibrated_linearsvc_isotonic",
    "calibrated_linearsvc_meta20",
    "calibrated_linearsvc_meta35",
    "calibrated_linearsvc_catboost20",
    "calibrated_linearsvc_proto10",
    "calibrated_linearsvc_knn10",
    "calibrated_linearsvc_proto5_knn5",
    "oof_stack_logreg",
}

METADATA_PREFIX_FIELDS = (
    "user",
    "service",
    "component",
    "request_type",
    "criticality",
    "urgency",
    "priority",
    "service_class",
    "timezone",
)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def labels_for_view(contract: Mapping[str, Any], view: str) -> list[str]:
    category_summary = contract["category_summary"]
    if view == "top15":
        return [str(item["name"]) for item in category_summary["top15"]]
    if view == "full43":
        return [str(item["name"]) for item in category_summary["all_categories"]]
    raise ValueError(f"Unknown category view: {view}")


def validate_search_ids(request_ids: Iterable[str], protocol: Mapping[str, Any]) -> None:
    lockbox = {str(value) for value in protocol["internal_lockbox_request_ids"]}
    overlap = lockbox.intersection(str(value) for value in request_ids)
    if overlap:
        sample = ", ".join(sorted(overlap)[:5])
        raise ValueError(f"V5 INTERNAL lockbox IDs are forbidden during search: {sample}")


def planned_folds(
    protocol: Mapping[str, Any],
    *,
    repeats: Sequence[int] | None = None,
    folds: Sequence[int] | None = None,
) -> list[dict[str, Any]]:
    wanted_repeats = None if repeats is None else {int(value) for value in repeats}
    wanted_folds = None if folds is None else {int(value) for value in folds}
    result: list[dict[str, Any]] = []
    for item in protocol["folds"]:
        repeat = int(item["repeat"])
        fold = int(item["fold"])
        if wanted_repeats is not None and repeat not in wanted_repeats:
            continue
        if wanted_folds is not None and fold not in wanted_folds:
            continue
        validate_search_ids(item["train_request_ids"], protocol)
        validate_search_ids(item["validation_request_ids"], protocol)
        result.append(dict(item))
    return sorted(result, key=lambda item: (int(item["repeat"]), int(item["fold"])))


def deterministic_run_id(candidate: Mapping[str, Any]) -> str:
    payload = json.dumps(candidate, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
    stage = str(candidate.get("stage", "run")).replace("/", "-")
    return f"{stage}-{digest}"


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(str(value).split())


def build_metadata_prefix(row: Mapping[str, Any]) -> str:
    parts = []
    for field in METADATA_PREFIX_FIELDS:
        value = _clean_text(row.get(field))
        if value:
            parts.append(f"{field}: {value}")
    return "\n".join(parts)


def build_text(row: Mapping[str, Any], *, feature_mode: str) -> str:
    if feature_mode not in FEATURE_MODES:
        raise ValueError(f"Unknown feature mode: {feature_mode}")
    description = _clean_text(row.get("description"))
    if feature_mode in {"text_only", "text_only_fallback", "separate_metadata_text", "metadata_prior_text"}:
        return description
    if feature_mode == "metadata_only_fallback":
        prefix = build_metadata_prefix(row)
        return prefix
    prefix = build_metadata_prefix(row)
    if prefix and description:
        return f"{prefix}\n\ndescription: {description}"
    return prefix or description


def truncate_embeddings(matrix: np.ndarray, dimension: int) -> np.ndarray:
    values = np.asarray(matrix)
    if values.ndim != 2:
        raise ValueError("Embeddings must be a two-dimensional matrix")
    if dimension <= 0:
        raise ValueError("Embedding dimension must be positive")
    if dimension > values.shape[1]:
        raise ValueError(f"Requested dimension {dimension} exceeds embedding width {values.shape[1]}")
    return values[:, :dimension]


def candidate_quality_key(metrics: Mapping[str, Any]) -> tuple[float, float, float, float]:
    return (
        float(metrics.get("top1_accuracy", float("-inf"))),
        float(metrics.get("macro_f1", float("-inf"))),
        float(metrics.get("top3_accuracy", float("-inf"))),
        float(metrics.get("true_label_mrr", float("-inf"))),
    )


def operator_metrics(
    truth: Sequence[str],
    probabilities: np.ndarray,
    labels: Sequence[str],
) -> dict[str, Any]:
    if not truth:
        raise ValueError("Cannot score an empty prediction set")
    values = np.asarray(probabilities, dtype=float)
    if values.shape != (len(truth), len(labels)):
        raise ValueError("Probability matrix shape does not match truth/labels")
    row_sums = values.sum(axis=1)
    if np.any(~np.isfinite(values)) or np.any(values < 0) or np.any(row_sums <= 0):
        raise ValueError("Probabilities must be finite, non-negative, and have positive row sums")
    values = values / row_sums[:, None]
    label_index = {str(label): index for index, label in enumerate(labels)}
    if any(str(label) not in label_index for label in truth):
        raise ValueError("Truth contains a label outside the evaluation label set")

    order = np.argsort(-values, axis=1)
    predicted = [str(labels[int(row[0])]) for row in order]
    ranks: list[int] = []
    for row_index, true_label in enumerate(truth):
        true_index = label_index[str(true_label)]
        positions = np.flatnonzero(order[row_index] == true_index)
        ranks.append(int(positions[0]) + 1)

    top = {k: sum(rank <= k for rank in ranks) / len(ranks) for k in (1, 2, 3, 5)}
    per_class = {}
    truth_counter = Counter(str(value) for value in truth)
    predicted_counter = Counter(str(value) for value in predicted)
    for label in labels:
        value = str(label)
        per_class[value] = {
            "support": truth_counter[value],
            "predicted": predicted_counter[value],
            "f1": round(
                float(f1_score(truth, predicted, labels=[value], average=None, zero_division=0)[0]),
                6,
            ),
        }
    return {
        "rows": len(truth),
        "top1_accuracy": round(top[1], 6),
        "top2_accuracy": round(top[2], 6),
        "top3_accuracy": round(top[3], 6),
        "top5_accuracy": round(top[5], 6),
        "macro_f1": round(
            float(f1_score(truth, predicted, labels=list(labels), average="macro", zero_division=0)),
            6,
        ),
        "weighted_f1": round(
            float(f1_score(truth, predicted, labels=list(labels), average="weighted", zero_division=0)),
            6,
        ),
        "balanced_accuracy": round(float(balanced_accuracy_score(truth, predicted)), 6),
        "true_label_mrr": round(sum(1.0 / rank for rank in ranks) / len(ranks), 6),
        "rank_distribution": {
            "rank1": sum(rank == 1 for rank in ranks),
            "rank2": sum(rank == 2 for rank in ranks),
            "rank3": sum(rank == 3 for rank in ranks),
            "rank_gt3": sum(rank > 3 for rank in ranks),
        },
        "per_class": per_class,
    }


def _stage_by_name(config: Mapping[str, Any], stage_name: str) -> Mapping[str, Any]:
    for stage in config.get("stages", []):
        if str(stage.get("name")) == stage_name:
            return stage
    raise ValueError(f"Unknown experiment stage: {stage_name}")


def _configured_models(config: Mapping[str, Any], stage: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    if "model" in config:
        models = [config["model"]]
    else:
        models = list(config.get("models", []))
    requested = {str(value) for value in stage.get("model_ids", [])}
    if requested:
        models = [model for model in models if str(model.get("id")) in requested]
    if not models:
        raise ValueError("Experiment config has no models for this stage")
    return models


def build_stage_candidates(
    config: Mapping[str, Any],
    stage_name: str,
    *,
    protocol: Mapping[str, Any],
) -> list[dict[str, Any]]:
    stage = _stage_by_name(config, stage_name)
    fold_plan = planned_folds(
        protocol,
        repeats=[int(value) for value in stage.get("repeats", [0])],
        folds=[int(value) for value in stage.get("folds", [0])],
    )
    dimensions = [int(value) for value in stage.get("embedding_dims", [2048])]
    lengths = [int(value) for value in stage.get("max_lengths", [256])]
    instructions = [str(value) for value in stage.get("instructions", ["none"])]
    feature_modes = [str(value) for value in stage.get("feature_modes", ["text_only"])]
    heads = [str(value) for value in stage.get("heads", ["logreg"])]
    views = [str(value) for value in stage.get("views", ["top15"])]

    unknown_instructions = set(instructions) - set(INSTRUCTION_REGISTRY)
    unknown_modes = set(feature_modes) - FEATURE_MODES
    unknown_heads = set(heads) - HEAD_REGISTRY
    if unknown_instructions:
        raise ValueError(f"Unknown instructions: {sorted(unknown_instructions)}")
    if unknown_modes:
        raise ValueError(f"Unknown feature modes: {sorted(unknown_modes)}")
    if unknown_heads:
        raise ValueError(f"Unknown heads: {sorted(unknown_heads)}")

    candidates: list[dict[str, Any]] = []
    for model, view, feature_mode, instruction, dimension, max_length, head, fold in itertools.product(
        _configured_models(config, stage),
        views,
        feature_modes,
        instructions,
        dimensions,
        lengths,
        heads,
        fold_plan,
    ):
        candidate = {
            "stage": stage_name,
            "server": str(config.get("server", "unknown")),
            "model_id": str(model["id"]),
            "model_revision": model.get("revision"),
            "deployment_policy": model.get("deployment_policy", "normal"),
            "view": view,
            "feature_mode": feature_mode,
            "instruction": instruction,
            "embedding_dim": int(dimension),
            "max_length": int(max_length),
            "head": head,
            "repeat": int(fold["repeat"]),
            "fold": int(fold["fold"]),
            "seed": int(fold["seed"]),
        }
        candidate["run_id"] = deterministic_run_id(candidate)
        candidates.append(candidate)
    return sorted(candidates, key=lambda item: item["run_id"])
