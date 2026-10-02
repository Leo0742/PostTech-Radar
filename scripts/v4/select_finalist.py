from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
ART = ROOT / "artifacts/gpu_research_v4"


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _metrics(item: dict[str, Any]) -> dict[str, float]:
    return {key: float(value["mean"]) for key, value in item["stability"].items()}


def _rare_mean(item: dict[str, Any]) -> float:
    values = []
    for metrics in item["repeat_metrics"].values():
        value = metrics["support_bands"]["1-5"]["macro_f1"]
        if value is not None:
            values.append(float(value))
    return sum(values) / len(values) if values else 0.0


def _quality(top: dict[str, Any], full: dict[str, Any]) -> float:
    top_metrics = _metrics(top)
    full_metrics = _metrics(full)
    rare = _rare_mean(full)
    temporal = full.get("temporal", {}).get("metrics", {}).get("macro_f1", 0.0)
    stability_penalty = 0.05 * float(full["stability"]["macro_f1"]["std"])
    return (
        0.15 * top_metrics["accuracy"]
        + 0.2 * top_metrics["macro_f1"]
        + 0.05 * top_metrics["worst_class_f1"]
        + 0.25 * full_metrics["macro_f1"]
        + 0.15 * full_metrics["balanced_accuracy"]
        + 0.1 * rare
        + 0.1 * temporal
        - stability_penalty
    )


def main() -> None:
    coverage_path = ART / "final/candidate-coverage-audit.json"
    if not coverage_path.exists():
        raise RuntimeError("candidate coverage audit must complete before finalist selection")
    coverage = _load(coverage_path)
    if coverage.get("status") != "COMPLETE" or coverage.get("measured_family_count") != coverage.get(
        "planned_family_count"
    ):
        raise RuntimeError("candidate coverage audit is incomplete")
    if coverage.get("sealed_holdout_accessed") is not False:
        raise RuntimeError("coverage audit must prove the sealed holdout remained closed")
    compliance_flag = ART / "final/.synthetic_compliance_complete"
    deadline = time.monotonic() + 12 * 60 * 60
    while not compliance_flag.exists() and time.monotonic() < deadline:
        time.sleep(15)
    if not compliance_flag.exists():
        raise RuntimeError("capped-weight synthetic finalist training did not finish within 12 hours")
    synthetic_finalists_path = ART / "deep/synthetic-finalists.json"
    deadline = time.monotonic() + 12 * 60 * 60
    while not synthetic_finalists_path.exists() and time.monotonic() < deadline:
        time.sleep(15)
    if not synthetic_finalists_path.exists():
        raise RuntimeError("synthetic finalist training did not finish within 12 hours")
    synthetic_finalists = _load(synthetic_finalists_path)
    neural_summary_path = ART / "deep/neural-finalist-summary.json"
    deadline = time.monotonic() + 12 * 60 * 60
    while not neural_summary_path.exists() and time.monotonic() < deadline:
        time.sleep(15)
    if not neural_summary_path.exists():
        raise RuntimeError("neural finalist summary did not finish within 12 hours")
    sources = {
        "structured": (
            ART / "deep/structured-top15.json",
            ART / "deep/structured-full43.json",
            3.0,
            0.2,
        ),
        "bge-m3": (
            ART / "deep/bge-m3-top15.json",
            ART / "deep/bge-m3-full43.json",
            3.0,
            1.3,
        ),
        "qwen3-embedding-4b": (
            ART / "deep/qwen3-embedding-4b-top15.json",
            ART / "deep/qwen3-embedding-4b-full43.json",
            34.0,
            8.6,
        ),
        "deep-oof-stack": (
            ART / "deep/stack-top15.json",
            ART / "deep/stack-full43.json",
            40.0,
            10.1,
        ),
    }
    candidates = []
    for family, (top_path, full_path, latency_ms, vram_gib) in sources.items():
        top = _load(top_path)
        full = _load(full_path)
        quality = _quality(top, full)
        cost_penalty = 0.012 * math.log1p(latency_ms) + 0.008 * math.log1p(vram_gib)
        candidates.append(
            {
                "family": family,
                "top15_candidate_id": top["candidate_id"],
                "full43_candidate_id": full["candidate_id"],
                "quality_score": round(quality, 8),
                "cost_penalty": round(cost_penalty, 8),
                "production_score": round(quality - cost_penalty, 8),
                "latency_ms_screening": latency_ms,
                "vram_gib_screening": vram_gib,
                "top15": _metrics(top),
                "full43": _metrics(full),
                "full43_temporal": full.get("temporal", {}).get("metrics"),
                "full43_support_bands": full["repeat_metrics"]["0"]["support_bands"],
                "full43_rare_1_5_macro_f1_mean": round(_rare_mean(full), 6),
                "full43_macro_f1_std": float(full["stability"]["macro_f1"]["std"]),
                "parameters": full.get("selected", {}).get("parameters"),
                "synthetic_recipe": (
                    synthetic_finalists["families"][family]["best"]
                    if family in synthetic_finalists["families"]
                    and synthetic_finalists["families"][family]["accepted"]
                    else None
                ),
            }
        )
    research = max(candidates, key=lambda item: item["quality_score"])
    deployable = [item for item in candidates if item["family"] != "deep-oof-stack"]
    production = max(deployable, key=lambda item: item["production_score"])
    synthetic = _load(ART / "deep/synthetic-recheck-full43.json")
    neural_finalists = _load(neural_summary_path)
    result = {
        "status": "FROZEN_BEFORE_SEALED_HOLDOUT",
        "selection_uses_sealed_holdout": False,
        "research_quality_winner": research,
        "production_winner": production,
        "synthetic_recipe": production.get("synthetic_recipe"),
        "synthetic_research": {
            "structured_initial_grid": synthetic,
            "family_specific": synthetic_finalists,
        },
        "neural_finalists": neural_finalists,
        "candidate_coverage_audit": "artifacts/gpu_research_v4/final/candidate-coverage-audit.json",
        "candidates": sorted(candidates, key=lambda item: item["production_score"], reverse=True),
        "policy": (
            "Weighted TOP15/FULL43/rare/temporal score, then an explicit local latency/VRAM penalty. "
            "The sealed holdout is not read until this file is persisted."
            " The OOF stack may be the research winner, but the exported production winner is selected only from "
            "independently deployable single-model families."
        ),
    }
    destination = ART / "final/selection.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
