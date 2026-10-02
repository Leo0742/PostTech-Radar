from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

from app.core.config import DATABASE_PATH  # noqa: E402
from app.ml.v4.evaluation import classification_metrics  # noqa: E402
from app.services.data_service import load_rows  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark label-semantic NLI classification")
    parser.add_argument("--model-id", default="MoritzLaurer/mDeBERTa-v3-base-mnli-xnli")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--view", choices=("top15", "full43"), required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--cache-dir", type=Path, default=Path(os.environ.get("HF_HOME", "/workspace/hf-cache")))
    args = parser.parse_args()

    import torch
    from huggingface_hub import model_info
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    pointer = json.loads((PROJECT_ROOT / "artifacts/gpu_research_v4/protocol.json").read_text(encoding="utf-8"))
    protocol = json.loads((PROJECT_ROOT / pointer["protocol_path"]).read_text(encoding="utf-8"))
    definitions = yaml.safe_load((PROJECT_ROOT / "data/category_definitions_v4.yaml").read_text(encoding="utf-8"))
    definition_by_label = {
        str(item["category"]): f"{item['definition']} Типичные термины: {', '.join(item['typical_symptoms'][:8])}."
        for item in definitions
    }
    all_rows = [row for row in load_rows(DATABASE_PATH) if str(row.get("description") or "").strip()]
    counts = Counter(str(row["category"]) for row in all_rows)
    labels = sorted({label for label, _ in counts.most_common(15)} if args.view == "top15" else set(counts))
    development = set(protocol["development_request_ids"])
    rows = [
        row
        for row in all_rows
        if str(row["request_id"]) in development and str(row["category"]) in set(labels)
    ]

    info = model_info(args.model_id, revision=args.revision)
    revision = str(info.sha)
    card = info.card_data.to_dict() if hasattr(info.card_data, "to_dict") else dict(info.card_data or {})
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, revision=revision, cache_dir=str(args.cache_dir))
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_id,
        revision=revision,
        cache_dir=str(args.cache_dir),
        torch_dtype=torch.bfloat16,
    ).cuda().eval()
    entailment_id = next(
        (index for name, index in model.config.label2id.items() if "entail" in name.lower()),
        0,
    )
    pairs = [
        (str(row["description"]), definition_by_label[label])
        for row in rows
        for label in labels
    ]
    scores: list[float] = []
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    with torch.inference_mode():
        for start in range(0, len(pairs), args.batch_size):
            chunk = pairs[start : start + args.batch_size]
            encoded = tokenizer(
                [item[0] for item in chunk],
                [item[1] for item in chunk],
                padding=True,
                truncation=True,
                max_length=256,
                return_tensors="pt",
            ).to("cuda")
            logits = model(**encoded).logits.float()
            scores.extend(torch.softmax(logits, dim=1)[:, entailment_id].cpu().tolist())
    seconds = time.perf_counter() - started
    matrix = np.asarray(scores).reshape(len(rows), len(labels))
    predicted = [labels[index] for index in matrix.argmax(axis=1)]
    truth = [str(row["category"]) for row in rows]
    payload: dict[str, Any] = {
        "candidate_id": f"category/mdeberta-nli/{args.view}/v4",
        "status": "MEASURED_GPU_LABEL_SEMANTIC",
        "model_id": args.model_id,
        "revision": revision,
        "license": str(card.get("license") or "UNKNOWN"),
        "trust_remote_code": False,
        "view": args.view,
        "dataset_sha256": protocol["dataset_sha256"],
        "split_sha256": protocol["split_sha256"],
        "sealed_holdout_accessed": False,
        "real_only_evaluation": True,
        "rows": len(rows),
        "label_pairs": len(pairs),
        "seconds": round(seconds, 3),
        "latency_ms_per_ticket": round(1000 * seconds / len(rows), 4),
        "peak_vram_bytes": int(torch.cuda.max_memory_allocated()),
        "metrics": classification_metrics(truth, predicted, labels=labels),
        "predictions": [
            {"request_id": str(row["request_id"]), "truth": truth[index], "prediction": predicted[index]}
            for index, row in enumerate(rows)
        ],
    }
    output = PROJECT_ROOT / "artifacts/gpu_research_v4/nli" / f"mdeberta-{args.view}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "metrics": payload["metrics"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
