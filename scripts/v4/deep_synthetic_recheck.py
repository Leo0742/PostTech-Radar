from __future__ import annotations

import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

from app.core.config import DATABASE_PATH  # noqa: E402
from app.ml.v3.candidates import CandidateSpec, build_estimator  # noqa: E402
from app.ml.v4.evaluation import classification_metrics  # noqa: E402
from app.services.data_service import load_rows  # noqa: E402


def build_synthetic_rows(
    source_samples: list[dict[str, Any]],
    approved_categories: set[str],
    rejected_hashes: set[str],
    prototypes: dict[str, dict[str, str]],
) -> list[dict[str, Any]]:
    return [
        {
            **prototypes.get(str(sample["target_category"]), {}),
            "request_id": f"synthetic-deep-{index}",
            "description": str(sample["description"]),
            "category": str(sample["target_category"]),
            "is_synthetic": True,
            "synthetic_sample_weight": float(sample.get("synthetic_sample_weight", 0.2)),
        }
        for index, sample in enumerate(source_samples)
        if str(sample["target_category"]) in approved_categories
        and str(sample["sample_sha256"]) not in rejected_hashes
    ]


def main() -> None:
    pointer = json.loads((PROJECT_ROOT / "artifacts/gpu_research_v4/protocol.json").read_text())
    protocol = json.loads((PROJECT_ROOT / pointer["protocol_path"]).read_text())
    fold = next(item for item in protocol["folds"] if int(item["repeat"]) == 0 and int(item["fold"]) == 0)
    rows = load_rows(DATABASE_PATH)
    by_id = {str(row["request_id"]): row for row in rows}
    train = [by_id[item] for item in fold["train_request_ids"]]
    validation = [by_id[item] for item in fold["validation_request_ids"]]
    support = Counter(str(row["category"]) for row in train)
    labels = sorted(set(protocol["labels"]))
    clean_path = PROJECT_ROOT / "artifacts/gpu_research_v4/synthetic/domain_corpus_clean.jsonl"
    deadline = time.monotonic() + 12 * 60 * 60
    while not clean_path.exists() and time.monotonic() < deadline:
        time.sleep(15)
    if not clean_path.exists():
        raise RuntimeError("clean synthetic corpus was not produced within 12 hours")
    source_samples = [json.loads(line) for line in clean_path.read_text(encoding="utf-8").splitlines() if line]
    audit_path = PROJECT_ROOT / "artifacts/gpu_research_v4/synthetic/manual_synthetic_audit.json"
    deadline = time.monotonic() + 12 * 60 * 60
    while not audit_path.exists() and time.monotonic() < deadline:
        time.sleep(15)
    if not audit_path.exists():
        raise RuntimeError("manual semantic synthetic audit was not completed within 12 hours")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit.get("status") != "COMPLETE":
        raise RuntimeError("manual semantic synthetic audit is not COMPLETE")
    approved_categories = set(audit.get("approved_categories", []))
    rejected_hashes = set(audit.get("rejected_sample_sha256", []))
    selected = json.loads(
        (PROJECT_ROOT / "artifacts/gpu_research_v4/deep/structured-full43.json").read_text(
            encoding="utf-8"
        )
    )["selected"]["parameters"]
    prototypes: dict[str, dict[str, str]] = {}
    fields = ("service", "component", "request_type", "criticality", "urgency", "priority", "service_class", "timezone")
    for label in labels:
        candidates = [row for row in train if str(row["category"]) == label]
        if candidates:
            signature = Counter(tuple((field, str(row.get(field) or "")) for field in fields) for row in candidates)
            prototypes[label] = dict(signature.most_common(1)[0][0])
    synthetic = build_synthetic_rows(source_samples, approved_categories, rejected_hashes, prototypes)
    truth = [str(row["category"]) for row in validation]
    grid: list[dict[str, Any]] = []
    for ratio in (5, 10, 25, 50, 100):
        chosen = []
        for label in labels:
            candidates = [row for row in synthetic if row["category"] == label]
            chosen.extend(candidates[: min(len(candidates), support[label] * ratio)])
        for weight in (0.0, 0.5, 1.0, 1.5, 2.0):
            model = build_estimator(
                CandidateSpec(
                    "category/deep-synthetic-grid/v4",
                    "category",
                    "structured_lr",
                    "combined",
                    parameters=selected,
                )
            )
            training = [*train, *chosen]
            sample_weight = np.asarray(
                [1.0] * len(train)
                + [weight * float(row["synthetic_sample_weight"]) for row in chosen]
            )
            model.fit(training, [str(row["category"]) for row in training], classifier__sample_weight=sample_weight)
            predicted = [str(value) for value in model.predict(validation)]
            grid.append(
                {
                    "ratio": ratio,
                    "weight_scale": weight,
                    "synthetic_rows": len(chosen),
                    "metrics": classification_metrics(truth, predicted, labels=labels),
                }
            )
    baseline = next(item for item in grid if item["ratio"] == 5 and item["weight_scale"] == 0.0)
    contenders = [item for item in grid if item["weight_scale"] > 0]
    best = max(
        contenders,
        key=lambda item: (
            item["metrics"]["macro_f1"],
            item["metrics"]["support_bands"]["1-5"]["macro_f1"] or 0.0,
        ),
    )
    baseline_rare = baseline["metrics"]["support_bands"]["1-5"]["macro_f1"] or 0.0
    best_rare = best["metrics"]["support_bands"]["1-5"]["macro_f1"] or 0.0
    survives = best["metrics"]["macro_f1"] > baseline["metrics"]["macro_f1"] and best_rare >= baseline_rare
    result = {
        "candidate_id": "category/deep-controlled-synthetic/full43/v4",
        "status": "ACCEPTED" if survives else "REJECTED_NO_REAL_VALIDATION_GAIN",
        "dataset_sha256": protocol["dataset_sha256"],
        "split_sha256": protocol["split_sha256"],
        "train_may_include_synthetic": True,
        "validation_real_only": True,
        "validation_rows": len(validation),
        "synthetic_source_rows": len(synthetic),
        "manual_semantic_audit": str(audit_path.relative_to(PROJECT_ROOT)),
        "manual_semantic_audit_summary": audit.get("summary"),
        "selected_real_only_parameters": selected,
        "baseline": baseline,
        "best": best,
        "grid": grid,
    }
    destination = PROJECT_ROOT / "artifacts/gpu_research_v4/deep/synthetic-recheck-full43.json"
    destination.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(destination), "status": result["status"], "baseline": baseline, "best": best}, ensure_ascii=False))


if __name__ == "__main__":
    main()
