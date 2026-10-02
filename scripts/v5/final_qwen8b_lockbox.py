from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.v5_dataset import DEFAULT_DATASET, load_v5_rows
from scripts.v5_experiment import DEFAULT_CONTRACT, DEFAULT_PROTOCOL, labels_for_view, load_json
from scripts.v5_gpu_runner import (
    DiskEmbeddingCache,
    SentenceTransformerBackend,
    evaluate_candidate_with_embedder,
    load_completed_stage_results,
    select_promoted_candidates,
)


DEFAULT_RESULTS = ROOT / "artifacts" / "gpu_research_v5" / "runs" / "qwen8b_quality_sprint"
DEFAULT_CACHE = ROOT / "artifacts" / "gpu_research_v5" / "embedding_cache"
DEFAULT_OUTPUT = ROOT / "outputs" / "V5_QWEN8B_FINAL_LOCKBOX.json"


def _supported_macro_f1(result: dict[str, Any]) -> dict[str, Any]:
    per_class = result["metrics"]["per_class"]
    labels = [label for label, values in per_class.items() if int(values["support"]) > 0]
    scores = [float(per_class[label]["f1"]) for label in labels]
    return {
        "labels": labels,
        "label_count": len(labels),
        "macro_f1": round(statistics.fmean(scores), 6) if scores else None,
    }


def _stage_is_complete(results_root: Path, stage: str) -> dict[str, Any]:
    path = Path(results_root) / stage / "stage_summary.json"
    if not path.exists():
        raise RuntimeError(f"Refusing to open the lockbox: {stage} has no completed stage summary")
    summary = json.loads(path.read_text(encoding="utf-8"))
    if int(summary.get("failed", 0)) != 0:
        raise RuntimeError(f"Refusing to open the lockbox: {stage} has failed runs")
    if int(summary.get("completed", -1)) + int(summary.get("skipped", 0)) != int(summary.get("candidate_count", -2)):
        raise RuntimeError(f"Refusing to open the lockbox: {stage} is incomplete")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="One-time final V5 INTERNAL LOCKBOX evaluation")
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--champion-stage", default="quality_ensemble_repeated")
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--unlock-final-lockbox", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if not args.unlock_final_lockbox:
        raise SystemExit("Pass --unlock-final-lockbox only after model selection is frozen.")
    if args.output.exists() and not args.force:
        raise SystemExit(f"Final lockbox artifact already exists: {args.output}")

    stage_summary = _stage_is_complete(args.results_root, args.champion_stage)
    previous = load_completed_stage_results(args.results_root, args.champion_stage)
    promoted = select_promoted_candidates(previous, top_n=1)
    if not promoted:
        raise RuntimeError("No completed champion candidate is available")
    champion = dict(promoted[0]["candidate"])
    if int(promoted[0]["fold_results"]) != 12:
        raise RuntimeError("Champion is not backed by all 12 frozen repeated folds")

    protocol = load_json(args.protocol)
    contract = load_json(args.contract)
    rows = load_v5_rows(args.dataset)
    train_ids = [
        *[str(value) for value in protocol["development_request_ids"]],
        *[str(value) for value in protocol["calibration_request_ids"]],
    ]
    lockbox_ids = [str(value) for value in protocol["internal_lockbox_request_ids"]]
    if set(train_ids).intersection(lockbox_ids):
        raise RuntimeError("Final train IDs overlap the internal lockbox")

    backend = SentenceTransformerBackend(batch_size=args.batch_size)
    embedder = DiskEmbeddingCache(args.cache_root, backend)
    evaluations: dict[str, Any] = {}
    for view in ("top15", "full43"):
        candidate = dict(champion)
        candidate.update(
            {
                "stage": "final_lockbox",
                "view": view,
                "repeat": 0,
                "fold": 0,
            }
        )
        result = evaluate_candidate_with_embedder(
            candidate,
            rows,
            train_ids=train_ids,
            validation_ids=lockbox_ids,
            labels=labels_for_view(contract, view),
            embedder=embedder,
        )
        result["internal_lockbox_accessed"] = True
        result["supported_lockbox_macro_f1"] = _supported_macro_f1(result)
        evaluations[view] = result

    payload = {
        "status": "FINAL_LOCKBOX_OPENED",
        "warning": "This artifact is final evaluation evidence and must not be used for further model selection.",
        "champion_source_stage": args.champion_stage,
        "champion_fold_results": promoted[0]["fold_results"],
        "champion_search_metrics": promoted[0]["aggregate_metrics"],
        "champion": champion,
        "stage_summary": {
            "candidate_count": stage_summary.get("candidate_count"),
            "completed": stage_summary.get("completed"),
            "skipped": stage_summary.get("skipped"),
            "failed": stage_summary.get("failed"),
        },
        "protocol": {
            "protocol_version": protocol.get("protocol_version"),
            "dataset_sha256": protocol.get("dataset_sha256"),
            "split_sha256": protocol.get("split_sha256"),
            "train_rows_before_view_filter": len(train_ids),
            "lockbox_rows_before_view_filter": len(lockbox_ids),
        },
        "evaluations": evaluations,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
