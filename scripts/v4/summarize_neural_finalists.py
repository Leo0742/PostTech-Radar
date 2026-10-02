from __future__ import annotations

import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
ART = ROOT / "artifacts/gpu_research_v4"
METRIC_NAMES = ("accuracy", "macro_f1", "balanced_accuracy", "worst_class_f1")


def summarize_metrics(runs: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    result = {}
    for name in METRIC_NAMES:
        values = [float(run[name]) for run in runs]
        result[name] = {
            "mean": round(statistics.fmean(values), 6),
            "std": round(statistics.pstdev(values), 6),
        }
    return result


def _selected_metrics(payload: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    metrics = payload["metrics"]
    if "accuracy" in metrics:
        return "fastfit", metrics
    head, selected = max(metrics.items(), key=lambda item: float(item[1]["macro_f1"]))
    return str(head), selected


def _valid_training_run(payload: dict[str, Any]) -> bool:
    if payload.get("mode") != "lora":
        return True
    return bool(payload.get("training_integrity", {}).get("input_require_grads_enabled"))


def main() -> None:
    paths = [
        *sorted((ART / "fastfit").glob("*.json")),
        *sorted((ART / "finetune").glob("*.json")),
    ]
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    excluded = []
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("status") != "MEASURED_GPU_FROZEN_FOLD":
            continue
        if not _valid_training_run(payload):
            excluded.append(
                {
                    "artifact": str(path.relative_to(ROOT)),
                    "reason": "LoRA + gradient checkpointing run did not record input_require_grads_enabled",
                }
            )
            continue
        head, metrics = _selected_metrics(payload)
        groups[str(payload["candidate_id"])].append(
            {
                "artifact": str(path.relative_to(ROOT)),
                "model_id": payload["model_id"],
                "view": payload["view"],
                "fold": int(payload["fold"]),
                "seed": int(payload["seed"]),
                "mode": payload.get("mode", "fastfit"),
                "selected_head": head,
                "hyperparameters": payload.get("hyperparameters", {"legacy_screening_artifact": True}),
                "metrics": metrics,
                "peak_vram_bytes": int(payload.get("peak_vram_bytes", 0)),
            }
        )
    summaries = {}
    for candidate_id, runs in sorted(groups.items()):
        summaries[candidate_id] = {
            "run_count": len(runs),
            "distinct_folds": sorted({run["fold"] for run in runs}),
            "distinct_seeds": sorted({run["seed"] for run in runs}),
            "stability": summarize_metrics([run["metrics"] for run in runs]),
            "best_run": max(runs, key=lambda run: float(run["metrics"]["macro_f1"])),
            "runs": runs,
        }
    result = {
        "status": "MEASURED_NEURAL_FINALISTS_FROZEN_DEVELOPMENT",
        "sealed_holdout_accessed": False,
        "selection_note": (
            "These finalists use multiple frozen folds/seeds but not the complete 12-fold confirmation. "
            "They are reported as neural-family evidence and are not eligible for automatic production promotion."
        ),
        "excluded_training_integrity_runs": excluded,
        "candidates": summaries,
    }
    destination = ART / "deep/neural-finalist-summary.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(destination), "candidates": len(summaries)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
