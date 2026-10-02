from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from statistics import mean
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.v5_gpu_runner import load_completed_stage_results


BASE = ROOT / "artifacts" / "gpu_research_v5"


def _mean(records: list[dict[str, Any]], metric: str) -> float:
    return round(mean(float(record["metrics"][metric]) for record in records), 6)


def _metrics(root: Path) -> dict[str, Any]:
    records = load_completed_stage_results(root, "matched_eval")
    top15 = [record for record in records if record["candidate"]["view"] == "top15"]
    full43 = [record for record in records if record["candidate"]["view"] == "full43"]
    if len(top15) != 12 or len(full43) != 12:
        raise RuntimeError(f"{root}: expected 12 TOP15 + 12 FULL43 results, got {len(top15)} + {len(full43)}")
    return {
        "top1": _mean(top15, "top1_accuracy"),
        "top3": _mean(top15, "top3_accuracy"),
        "macro_f1": _mean(top15, "macro_f1"),
        "full43_macro_f1": _mean(full43, "macro_f1"),
    }


def _benchmark(root: Path) -> dict[str, Any]:
    records = load_completed_stage_results(root, "matched_benchmark")
    if len(records) != 1:
        raise RuntimeError(f"{root}: expected one matched benchmark result, got {len(records)}")
    result = records[0]
    runtime = result["runtime"]
    calls = list(runtime.get("embedding_backend_call_stats", []))
    if calls:
        backend = max(calls, key=lambda item: int(item.get("rows", 0) or 0))
        peak_bytes = max(int(item.get("peak_vram_bytes", 0) or 0) for item in calls)
        total_embed_seconds = sum(float(item.get("seconds", 0.0) or 0.0) for item in calls)
    else:
        backend = runtime.get("embedding_backend_last_stats", {})
        peak_bytes = int(backend.get("peak_vram_bytes", 0) or 0)
        total_embed_seconds = float(backend.get("seconds", 0.0) or 0.0)
    rows = int(backend.get("rows", 0) or 0)
    embed_seconds = float(backend.get("seconds", 0.0) or 0.0)
    if rows <= 0 or embed_seconds <= 0 or peak_bytes <= 0:
        raise RuntimeError(f"{root}: benchmark did not record a fresh embedding pass with latency/VRAM stats")
    return {
        "end_to_end_seconds": round(float(runtime["seconds"]), 6),
        "total_embedding_seconds": round(total_embed_seconds, 6),
        "embedding_seconds": round(embed_seconds, 6),
        "embedding_ms_per_row": round(1000.0 * embed_seconds / rows, 3),
        "peak_vram_mib": round(peak_bytes / (1024 * 1024), 1),
        "embedding_rows": rows,
        "used_batch_size": backend.get("used_batch_size"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Report the matched Qwen4B vs Qwen8B V5 reference.")
    parser.add_argument("--manifest", type=Path, default=BASE / "matched_reference" / "frozen_recipe.json")
    parser.add_argument("--json-output", type=Path, default=ROOT / "outputs" / "MATCHED_QWEN4B_VS_QWEN8B_V5.json")
    parser.add_argument("--md-output", type=Path, default=ROOT / "outputs" / "MATCHED_QWEN4B_VS_QWEN8B_V5.md")
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    q8 = {
        **_metrics(BASE / "runs" / "matched_qwen8b"),
        **_benchmark(BASE / "runs" / "matched_benchmark_qwen8b"),
    }
    q4 = {
        **_metrics(BASE / "runs" / "matched_qwen4b"),
        **_benchmark(BASE / "runs" / "matched_benchmark_qwen4b"),
    }
    delta = {
        key: round(float(q8[key]) - float(q4[key]), 6)
        for key in ("top1", "top3", "macro_f1", "full43_macro_f1")
    }
    report = {
        "status": "complete",
        "matched_recipe": manifest["matched_recipe"],
        "qwen4b": q4,
        "qwen8b": q8,
        "qwen8b_minus_qwen4b": delta,
        "latency_protocol": "same TOP15 repeat=0 fold=0, fresh dedicated embedding cache, batch size 8",
        "vram_protocol": "torch CUDA peak allocated memory during the same fresh embedding pass",
        "fairness_note": (
            "The matched pair uses the largest common embedding dimension. "
            "If the frozen 8B winner uses a wider vector than Qwen4B supports, both matched models are truncated to Qwen4B's native width."
        ),
    }
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    r = manifest["matched_recipe"]
    lines = [
        "# Qwen4B matched V5 recipe vs Qwen8B matched V5 recipe",
        "",
        "This is a matched-model-size reference, not a re-tuning of Qwen4B.",
        "",
        f"- Instruction: {r['instruction']}",
        f"- Representation: {r['feature_mode']}",
        f"- Sequence length: {r['max_length']}",
        f"- Common embedding dimension: {r['embedding_dim']}",
        f"- Frozen 8B winning dimension before matching: {r['winning_qwen8b_embedding_dim']}",
        f"- Head/calibration: {r['head']} / {r['calibration']}",
        f"- Synthetic recipe: {r['synthetic_recipe']}",
        "",
        "| Metric | Qwen4B matched | Qwen8B matched | 8B - 4B |",
        "|---|---:|---:|---:|",
        f"| TOP1 | {q4['top1']:.6f} | {q8['top1']:.6f} | {delta['top1']:+.6f} |",
        f"| TOP3 | {q4['top3']:.6f} | {q8['top3']:.6f} | {delta['top3']:+.6f} |",
        f"| Macro-F1 | {q4['macro_f1']:.6f} | {q8['macro_f1']:.6f} | {delta['macro_f1']:+.6f} |",
        f"| FULL43 Macro-F1 | {q4['full43_macro_f1']:.6f} | {q8['full43_macro_f1']:.6f} | {delta['full43_macro_f1']:+.6f} |",
        f"| Benchmark end-to-end, s | {q4['end_to_end_seconds']:.3f} | {q8['end_to_end_seconds']:.3f} | — |",
        f"| Embedding latency, ms/row | {q4['embedding_ms_per_row']:.3f} | {q8['embedding_ms_per_row']:.3f} | — |",
        f"| Peak VRAM, MiB | {q4['peak_vram_mib']:.1f} | {q8['peak_vram_mib']:.1f} | — |",
        "",
        "Latency/VRAM benchmark: same TOP15 repeat 0 / fold 0, fresh dedicated cache, batch size 8.",
    ]
    args.md_output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
