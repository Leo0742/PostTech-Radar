from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS = ROOT / "artifacts" / "gpu_research_v5" / "runs" / "qwen8b_quality_sprint"
DEFAULT_JSON = ROOT / "outputs" / "V5_QWEN8B_QUALITY_SPRINT.json"
DEFAULT_MD = ROOT / "outputs" / "V5_QWEN8B_QUALITY_SPRINT.md"
IDENTITY_EXCLUDE = {"stage", "repeat", "fold", "seed", "run_id", "server"}
METRICS = ("top1_accuracy", "macro_f1", "top3_accuracy", "true_label_mrr")


def _load_results(stage_root: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if not stage_root.exists():
        return records
    for path in sorted(stage_root.glob("*/result.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if payload.get("status") == "complete" and isinstance(payload.get("candidate"), Mapping):
            payload["_path"] = str(path)
            records.append(payload)
    return records


def _identity(candidate: Mapping[str, Any]) -> str:
    payload = {key: value for key, value in candidate.items() if key not in IDENTITY_EXCLUDE}
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _stat(values: Sequence[float]) -> dict[str, Any]:
    clean = [float(value) for value in values]
    return {
        "mean": round(statistics.fmean(clean), 6) if clean else None,
        "std": round(statistics.pstdev(clean), 6) if len(clean) > 1 else (0.0 if clean else None),
        "n": len(clean),
    }


def _aggregate_stage(results_root: Path, stage: str) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for result in _load_results(results_root / stage):
        grouped[_identity(result["candidate"])].append(result)

    rows: list[dict[str, Any]] = []
    for records in grouped.values():
        candidate = {
            key: value
            for key, value in records[0]["candidate"].items()
            if key not in {"repeat", "fold", "seed", "run_id"}
        }
        metrics = {
            metric: _stat(
                [record["metrics"][metric] for record in records if metric in record.get("metrics", {})]
            )
            for metric in METRICS
        }

        evidence: dict[str, dict[str, Any]] = {}
        for band in ("zero_real_train", "single_real_anchor", "two_to_five_real"):
            values = [
                record.get("training_evidence_bands", {}).get(band, {}).get("macro_f1")
                for record in records
            ]
            evidence[band] = _stat([value for value in values if value is not None])

        confusions: Counter[tuple[str, str]] = Counter()
        for record in records:
            for prediction in record.get("predictions", []):
                truth = str(prediction.get("truth"))
                predicted = str(prediction.get("top1"))
                if truth != predicted:
                    confusions[(truth, predicted)] += 1

        rows.append(
            {
                "candidate": candidate,
                "fold_results": len(records),
                "metrics": metrics,
                "evidence_bands": evidence,
                "top_confusions": [
                    {"truth": truth, "predicted": predicted, "count": count}
                    for (truth, predicted), count in confusions.most_common(10)
                ],
            }
        )

    def quality_key(item: Mapping[str, Any]) -> tuple[float, float, float, float]:
        metrics = item["metrics"]
        return tuple(float(metrics[key]["mean"] or 0.0) for key in METRICS)

    rows.sort(key=quality_key, reverse=True)
    return rows


def _pct(value: Any) -> str:
    return "—" if value is None else f"{100.0 * float(value):.2f}%"


def _markdown(report: Mapping[str, Any]) -> str:
    lines = ["# Qwen3-Embedding-8B V5 quality sprint", ""]
    for stage, rows in report["stages"].items():
        lines.extend(
            [
                f"## {stage}",
                "",
                "| Candidate | n | TOP1 | TOP3 | Macro-F1 | MRR | single-anchor F1 | 2–5 F1 |",
                "|---|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in rows:
            candidate = row["candidate"]
            label = ", ".join(
                str(value)
                for value in (
                    candidate.get("view"),
                    f"dim={candidate.get('embedding_dim')}",
                    candidate.get("instruction"),
                    candidate.get("head"),
                )
            )
            metrics = row["metrics"]
            evidence = row["evidence_bands"]
            lines.append(
                "| "
                + " | ".join(
                    [
                        label,
                        str(row["fold_results"]),
                        f"{_pct(metrics['top1_accuracy']['mean'])} ± {_pct(metrics['top1_accuracy']['std'])}",
                        f"{_pct(metrics['top3_accuracy']['mean'])} ± {_pct(metrics['top3_accuracy']['std'])}",
                        f"{_pct(metrics['macro_f1']['mean'])} ± {_pct(metrics['macro_f1']['std'])}",
                        f"{_pct(metrics['true_label_mrr']['mean'])} ± {_pct(metrics['true_label_mrr']['std'])}",
                        _pct(evidence["single_real_anchor"]["mean"]),
                        _pct(evidence["two_to_five_real"]["mean"]),
                    ]
                )
                + " |"
            )
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--stage", action="append", dest="stages")
    parser.add_argument("--output-json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_MD)
    args = parser.parse_args()

    stages = args.stages or [
        "quality_dim_repeated",
        "quality_instruction_repeated",
        "quality_head_repeated",
        "quality_head_refine",
        "quality_final_repeated",
    ]
    report = {
        "results_root": str(args.results_root),
        "stages": {stage: _aggregate_stage(args.results_root, stage) for stage in stages},
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    args.output_md.write_text(_markdown(report), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
