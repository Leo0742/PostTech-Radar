from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "backend"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from scripts.v5_dataset import DEFAULT_DATASET, load_v5_rows


def _row_payload(row: dict[str, object]) -> dict[str, object]:
    fields = (
        "description",
        "registration_date",
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
    return {field: row.get(field, "") for field in fields}


def _top3(pipeline: object, probabilities: np.ndarray) -> list[dict[str, object]]:
    classes = np.asarray(pipeline.classes_, dtype=object)
    order = np.argsort(probabilities)[::-1][:3]
    return [
        {"label": str(classes[index]), "confidence": round(float(probabilities[index]), 6)}
        for index in order
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark a serialized V5 category artifact on CUDA")
    parser.add_argument("artifact", type=Path)
    parser.add_argument("--runs", type=int, default=12)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    rows = load_v5_rows(DEFAULT_DATASET)
    bundle = joblib.load(args.artifact)
    pipeline = bundle["pipeline"]
    payloads = [_row_payload(row) for row in rows[: max(args.runs + 2, 16)]]

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the V5 benchmark")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    started = time.perf_counter()
    first_probabilities = np.asarray(pipeline.predict_proba([payloads[0]])[0], dtype=float)
    torch.cuda.synchronize()
    cold_seconds = time.perf_counter() - started

    warm_seconds: list[float] = []
    warm_top3: list[dict[str, object]] = []
    for payload in payloads[1 : args.runs + 1]:
        started = time.perf_counter()
        probabilities = np.asarray(pipeline.predict_proba([payload])[0], dtype=float)
        torch.cuda.synchronize()
        warm_seconds.append(time.perf_counter() - started)
        warm_top3 = _top3(pipeline, probabilities)

    partial_payload = {"description": str(payloads[-1].get("description") or "")}
    partial_probabilities = np.asarray(pipeline.predict_proba([partial_payload])[0], dtype=float)
    torch.cuda.synchronize()

    sorted_warm = sorted(warm_seconds)
    p95_index = max(0, min(len(sorted_warm) - 1, int(round(0.95 * (len(sorted_warm) - 1)))))
    result = {
        "artifact": str(args.artifact),
        "artifact_bytes": args.artifact.stat().st_size,
        "model_id": getattr(pipeline, "model_id", None),
        "model_revision": getattr(pipeline, "model_revision", None),
        "embedding_dim": getattr(pipeline, "embedding_dim", None),
        "metadata_scale": float(getattr(pipeline, "metadata_scale", 1.0)),
        "cold_inference_seconds": round(cold_seconds, 6),
        "warm_mean_seconds": round(statistics.mean(warm_seconds), 6),
        "warm_median_seconds": round(statistics.median(warm_seconds), 6),
        "warm_p95_seconds": round(sorted_warm[p95_index], 6),
        "warm_runs": len(warm_seconds),
        "peak_vram_mib": round(torch.cuda.max_memory_allocated() / (1024**2), 2),
        "first_probability_sum": round(float(first_probabilities.sum()), 9),
        "first_top3": _top3(pipeline, first_probabilities),
        "warm_top3": warm_top3,
        "partial_input_probability_sum": round(float(partial_probabilities.sum()), 9),
        "partial_input_top3": _top3(pipeline, partial_probabilities),
    }
    text = json.dumps(result, ensure_ascii=False, indent=2)
    print(text)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
