from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.v5_data_audit import read_xlsx_rows


DEFAULT_DATASET = PROJECT_ROOT / "data" / "raw" / "Обращения_1931.xlsx"
DEFAULT_PROTOCOL = PROJECT_ROOT / "artifacts" / "gpu_research_v5" / "protocol" / "protocol.json"
DEFAULT_CONTRACT = PROJECT_ROOT / "artifacts" / "gpu_research_v5" / "data_contract.json"
BASELINE_ROOT = PROJECT_ROOT / "artifacts" / "gpu_research_v5" / "baselines"
OUTPUT_ROOT = PROJECT_ROOT / "outputs"

MISSING = "__MISSING__"
METADATA_KEYS = (
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

SOURCE_TO_KEY = {
    "Пользователь": "user",
    "Услуга": "service",
    "Компонент услуги 1 уровня": "component",
    "Тип запроса": "request_type",
    "Критичность": "criticality",
    "Срочность": "urgency",
    "Приоритет": "priority",
    "Класс обслуживания": "service_class",
    "Часовой пояс запроса": "timezone",
}


def _request_id(value: Any) -> str:
    if value in {None, ""}:
        raise ValueError("Номер запроса cannot be empty")
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value).strip()


def _registration_datetime(value: Any) -> datetime:
    if value in {None, ""}:
        raise ValueError("Дата регистрации cannot be empty")
    if isinstance(value, (int, float)):
        return datetime(1899, 12, 30) + timedelta(days=float(value))
    return datetime.fromisoformat(str(value).strip().replace("Z", "+00:00")).replace(tzinfo=None)


def _clean(value: Any) -> str:
    if value in {None, ""}:
        return MISSING
    text = " ".join(str(value).split())
    return text or MISSING


def build_metadata_row(source: dict[str, Any]) -> dict[str, str]:
    result = {target: _clean(source.get(source_name)) for source_name, target in SOURCE_TO_KEY.items()}
    moment = _registration_datetime(source.get("Дата регистрации"))
    result.update(
        {
            "registration_month": str(moment.month),
            "registration_weekday": str(moment.weekday()),
            "registration_hour": str(moment.hour),
        }
    )
    return result


def load_source_rows(path: Path) -> list[dict[str, Any]]:
    _sheet, _headers, rows, _storage = read_xlsx_rows(Path(path))
    result = []
    for source in rows:
        result.append(
            {
                "request_id": _request_id(source["Номер запроса"]),
                "category": str(source["Вид запроса"] or "").strip(),
                "metadata": build_metadata_row(source),
            }
        )
    return result


def _matrix(rows: list[dict[str, Any]]) -> list[list[str]]:
    return [[row["metadata"][key] for key in METADATA_KEYS] for row in rows]


def _model(*, seed: int, c: float, class_weight: str | None, min_frequency: int = 2) -> Pipeline:
    return Pipeline(
        [
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    min_frequency=min_frequency,
                    sparse_output=True,
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=2500,
                    C=c,
                    class_weight=class_weight,
                    random_state=seed,
                ),
            ),
        ]
    )


def _align_probabilities(model: Pipeline, values: list[list[str]], labels: list[str]) -> np.ndarray:
    local = np.asarray(model.predict_proba(values), dtype=float)
    classes = [str(value) for value in model.named_steps["classifier"].classes_]
    result = np.zeros((len(values), len(labels)), dtype=float)
    target_index = {label: index for index, label in enumerate(labels)}
    for source_index, label in enumerate(classes):
        if label in target_index:
            result[:, target_index[label]] = local[:, source_index]
    row_sums = result.sum(axis=1, keepdims=True)
    np.divide(result, row_sums, out=result, where=row_sums > 0)
    return result


def _metrics(truth: list[str], probabilities: np.ndarray, labels: list[str]) -> dict[str, Any]:
    if not truth:
        raise ValueError("Cannot score an empty prediction set")
    label_index = {label: index for index, label in enumerate(labels)}
    order = np.argsort(-probabilities, axis=1)
    predicted = [labels[int(row[0])] for row in order]
    ranks: list[int] = []
    for index, true_label in enumerate(truth):
        true_index = label_index[true_label]
        positions = np.where(order[index] == true_index)[0]
        ranks.append(int(positions[0]) + 1 if len(positions) else len(labels) + 1)

    top = {
        k: sum(rank <= k for rank in ranks) / len(ranks)
        for k in (1, 2, 3, 5)
    }
    rank_distribution = {
        "rank1": sum(rank == 1 for rank in ranks),
        "rank2": sum(rank == 2 for rank in ranks),
        "rank3": sum(rank == 3 for rank in ranks),
        "rank_gt3": sum(rank > 3 for rank in ranks),
    }
    return {
        "rows": len(truth),
        "top1_accuracy": round(top[1], 6),
        "top2_accuracy": round(top[2], 6),
        "top3_accuracy": round(top[3], 6),
        "top5_accuracy": round(top[5], 6),
        "macro_f1": round(float(f1_score(truth, predicted, labels=labels, average="macro", zero_division=0)), 6),
        "weighted_f1": round(float(f1_score(truth, predicted, labels=labels, average="weighted", zero_division=0)), 6),
        "balanced_accuracy": round(float(balanced_accuracy_score(truth, predicted)), 6),
        "true_label_mrr": round(sum(1.0 / rank for rank in ranks) / len(ranks), 6),
        "rank_distribution": rank_distribution,
    }


def _evaluate_config(
    rows_by_id: dict[str, dict[str, Any]],
    folds: list[dict[str, Any]],
    labels: list[str],
    config: dict[str, Any],
    *,
    repeat_filter: int | None,
    keep_predictions_repeat0: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    truth_all: list[str] = []
    prob_all: list[np.ndarray] = []
    predictions_repeat0: list[dict[str, Any]] = []
    fold_metrics: list[dict[str, Any]] = []
    label_set = set(labels)

    for fold in folds:
        repeat = int(fold["repeat"])
        if repeat_filter is not None and repeat != repeat_filter:
            continue
        train = [
            rows_by_id[item]
            for item in fold["train_request_ids"]
            if item in rows_by_id and rows_by_id[item]["category"] in label_set
        ]
        validation = [
            rows_by_id[item]
            for item in fold["validation_request_ids"]
            if item in rows_by_id and rows_by_id[item]["category"] in label_set
        ]
        if not train or not validation:
            continue
        model = _model(
            seed=int(fold["seed"]),
            c=float(config["C"]),
            class_weight=config["class_weight"],
            min_frequency=int(config.get("min_frequency", 2)),
        )
        model.fit(_matrix(train), [row["category"] for row in train])
        probabilities = _align_probabilities(model, _matrix(validation), labels)
        truth = [row["category"] for row in validation]
        metrics = _metrics(truth, probabilities, labels)
        fold_metrics.append(
            {
                "repeat": repeat,
                "fold": int(fold["fold"]),
                "seed": int(fold["seed"]),
                "train": len(train),
                "validation": len(validation),
                "metrics": metrics,
            }
        )
        truth_all.extend(truth)
        prob_all.extend(probabilities)

        if keep_predictions_repeat0 and repeat == 0:
            for row, probs in zip(validation, probabilities, strict=True):
                predictions_repeat0.append(
                    {
                        "request_id": row["request_id"],
                        "truth": row["category"],
                        "probabilities": {
                            label: round(float(value), 10)
                            for label, value in zip(labels, probs, strict=True)
                        },
                    }
                )

    if not prob_all:
        raise ValueError("No metadata baseline predictions were produced")
    metrics = _metrics(truth_all, np.asarray(prob_all, dtype=float), labels)
    return metrics, fold_metrics, predictions_repeat0


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _labels_for_view(contract: dict[str, Any], view: str) -> list[str]:
    if view == "top15":
        return [item["name"] for item in contract["category_summary"]["top15"]]
    if view == "full43":
        return [item["name"] for item in contract["category_summary"]["all_categories"]]
    raise ValueError(f"Unknown view: {view}")


def _catboost_status() -> dict[str, Any]:
    try:
        import catboost  # type: ignore
    except ModuleNotFoundError:
        return {"available": False, "status": "not_available_local"}
    return {"available": True, "status": "available", "version": catboost.__version__}


def run_view(
    rows: list[dict[str, Any]],
    protocol_path: Path,
    contract_path: Path,
    *,
    view: str,
    quick: bool = False,
) -> dict[str, Any]:
    protocol = _load_json(protocol_path)
    contract = _load_json(contract_path)
    labels = _labels_for_view(contract, view)
    development_ids = set(protocol["development_request_ids"])
    rows_by_id = {row["request_id"]: row for row in rows if row["request_id"] in development_ids}
    folds = list(protocol["folds"])

    if quick:
        candidates = [{"C": 1.0, "class_weight": "balanced", "min_frequency": 2}]
    else:
        candidates = [
            {"C": c, "class_weight": class_weight, "min_frequency": 2}
            for c in (0.5, 1.0, 2.0)
            for class_weight in (None, "balanced")
        ]

    selection = []
    for config in candidates:
        metrics, _fold_metrics, _predictions = _evaluate_config(
            rows_by_id,
            folds,
            labels,
            config,
            repeat_filter=0,
            keep_predictions_repeat0=False,
        )
        selection.append({"config": config, "metrics": metrics})
    selected = max(
        selection,
        key=lambda item: (
            item["metrics"]["top1_accuracy"],
            item["metrics"]["macro_f1"],
            item["metrics"]["top3_accuracy"],
            item["metrics"]["true_label_mrr"],
        ),
    )

    repeat_filter = 0 if quick else None
    metrics, fold_metrics, predictions_repeat0 = _evaluate_config(
        rows_by_id,
        folds,
        labels,
        selected["config"],
        repeat_filter=repeat_filter,
        keep_predictions_repeat0=True,
    )
    repeat_summaries = []
    for repeat in sorted({int(item["repeat"]) for item in fold_metrics}):
        repeat_truth: list[str] = []
        repeat_probabilities: list[list[float]] = []
        predictions_by_id = {
            item["request_id"]: item
            for item in predictions_repeat0
        } if repeat == 0 else {}
        if repeat == 0 and predictions_by_id:
            for item in predictions_repeat0:
                repeat_truth.append(item["truth"])
                repeat_probabilities.append([item["probabilities"][label] for label in labels])
            repeat_summaries.append(
                {"repeat": repeat, "metrics": _metrics(repeat_truth, np.asarray(repeat_probabilities), labels)}
            )
        else:
            # Fold-level metrics are kept for other repeats; pooled metrics remain the stable comparison.
            repeat_summaries.append({"repeat": repeat, "fold_count": sum(f["repeat"] == repeat for f in fold_metrics)})

    return {
        "candidate_id": f"category/metadata_onehot_lr/{view}/v5.1",
        "view": view,
        "family": "missing_aware_onehot_logistic_regression",
        "feature_keys": list(METADATA_KEYS),
        "missing_token": MISSING,
        "date_policy": "registration date is reduced to month/weekday/hour categorical features; raw request ID is excluded",
        "selected_config": selected["config"],
        "development_selection": selection,
        "metrics": metrics,
        "fold_metrics": fold_metrics,
        "repeat_summaries": repeat_summaries,
        "oof_predictions_repeat0": sorted(predictions_repeat0, key=lambda item: item["request_id"]),
        "labels": labels,
        "protocol": {
            "dataset_sha256": protocol["dataset_sha256"],
            "split_sha256": protocol["split_sha256"],
            "internal_lockbox_request_ids": protocol["internal_lockbox_request_ids"],
        },
        "real_only_evaluation": True,
        "synthetic_rows_in_validation": 0,
        "internal_lockbox_accessed": False,
        "catboost": _catboost_status(),
    }


def _render_report(top15: dict[str, Any], full43: dict[str, Any]) -> str:
    lines = [
        "# PostTech Radar V5.1 — Metadata Baselines",
        "",
        "All metrics below are development-only grouped OOF results. V5 INTERNAL LOCKBOX remains untouched.",
        "",
        "| View | TOP1 | TOP3 | Macro-F1 | Balanced accuracy | MRR |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for payload in (top15, full43):
        metrics = payload["metrics"]
        lines.append(
            f"| {payload['view'].upper()} | {metrics['top1_accuracy']:.4f} | {metrics['top3_accuracy']:.4f} | {metrics['macro_f1']:.4f} | {metrics['balanced_accuracy']:.4f} | {metrics['true_label_mrr']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Feature contract",
            "",
            "- Separate metadata-only branch; description text is not used.",
            "- Missing values use an explicit `__MISSING__` category.",
            "- Registration date contributes month/weekday/hour only.",
            "- Request ID and all post-resolution fields are excluded.",
            "- CatBoost is recorded separately and is not silently substituted when unavailable.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="PostTech Radar V5.1 metadata-only grouped OOF baselines")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    args = parser.parse_args()

    rows = load_source_rows(args.dataset)
    top15 = run_view(rows, args.protocol, args.contract, view="top15")
    full43 = run_view(rows, args.protocol, args.contract, view="full43")
    BASELINE_ROOT.mkdir(parents=True, exist_ok=True)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    (BASELINE_ROOT / "metadata_top15.json").write_text(
        json.dumps(top15, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (BASELINE_ROOT / "metadata_full43.json").write_text(
        json.dumps(full43, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (OUTPUT_ROOT / "BASELINES_V5.md").write_text(_render_report(top15, full43), encoding="utf-8")
    print(
        json.dumps(
            {
                "top15": top15["metrics"],
                "full43": full43["metrics"],
                "catboost": top15["catboost"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
