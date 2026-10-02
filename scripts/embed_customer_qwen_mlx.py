from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import joblib
import mlx.core as mx
import numpy as np
from mlx_embeddings import load


ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

OUTPUT_DIR = ROOT / "outputs/customer_eval_2026-09-21"


def main() -> None:
    rows = json.loads((OUTPUT_DIR / "new_rows.json").read_text(encoding="utf-8"))
    bundle = joblib.load(ROOT / "models/v5/category_qwen4b_lite.joblib")
    pipeline = bundle["pipeline"]
    instruction = str(getattr(pipeline, "instruction", "") or "")
    texts = []
    for row in rows:
        description = " ".join(str(row.get("description") or "").split())
        texts.append(
            f"Instruct: {instruction}\nQuery: {description}" if instruction else description
        )

    model_dir = Path(
        os.getenv(
            "QWEN_MLX_MODEL_DIR",
            ROOT / "models/encoders/Qwen3-Embedding-4B-MLX-4bit",
        )
    )
    started = time.perf_counter()
    model, tokenizer = load(str(model_dir))
    embeddings: list[np.ndarray] = []
    batch_size = 4
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        inputs = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=int(getattr(pipeline, "max_length", 512)),
            return_tensors="mlx",
        )
        outputs = model(inputs["input_ids"], attention_mask=inputs["attention_mask"])
        mx.eval(outputs.text_embeds)
        embeddings.append(np.asarray(outputs.text_embeds, dtype=np.float32))
        print(f"embedded {min(start + batch_size, len(texts))}/{len(texts)}", flush=True)

    matrix = np.concatenate(embeddings, axis=0)
    output = OUTPUT_DIR / "qwen4b_new.npy"
    np.save(output, matrix)
    manifest = {
        "rows": len(rows),
        "dimension": int(matrix.shape[1]),
        "runtime": "MLX-4bit",
        "model_dir": str(model_dir),
        "source_model_id": bundle.get("metadata", {}).get("model_id"),
        "source_revision": bundle.get("metadata", {}).get("model_revision"),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }
    output.with_suffix(".json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
