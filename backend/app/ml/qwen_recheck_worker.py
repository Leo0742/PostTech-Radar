from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import joblib
import numpy as np

BACKEND_ROOT = Path(__file__).resolve().parents[2]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))


def _mlx_embeddings(pipeline: object, payload: dict[str, object]) -> np.ndarray:
    import mlx.core as mx
    from mlx_embeddings import load
    from mlx_lm.tuner.lora import LoRALinear

    model_dir = os.getenv("QWEN_MLX_MODEL_DIR", "").strip()
    if not model_dir:
        raise RuntimeError("QWEN_MLX_MODEL_DIR is not configured")

    description = " ".join(str(payload.get("description") or "").split())
    instruction = str(getattr(pipeline, "instruction", "") or "")
    text = f"Instruct: {instruction}\nQuery: {description}" if instruction else description
    model, tokenizer = load(model_dir, lazy=True)

    adapter_dir = str(getattr(pipeline, "adapter_dir", "") or "").strip()
    if adapter_dir:
        adapter_path = Path(adapter_dir)
        if not adapter_path.is_absolute():
            adapter_path = Path(__file__).resolve().parents[3] / adapter_path
        config_path = adapter_path / "adapter_config.json"
        weights_path = adapter_path / "adapter_model.safetensors"
        if not config_path.exists() or not weights_path.exists():
            raise RuntimeError(f"Qwen MLX adapter is incomplete: {adapter_path}")

        config = json.loads(config_path.read_text(encoding="utf-8"))
        rank = int(config.get("r", 0))
        alpha = float(config.get("lora_alpha", 0.0))
        targets = [str(value) for value in config.get("target_modules", [])]
        if rank <= 0 or alpha <= 0 or not targets:
            raise RuntimeError("Unsupported Qwen LoRA adapter configuration")

        adapter_weights = mx.load(str(weights_path))
        used_keys: set[str] = set()
        scale = alpha / rank
        for layer_index, layer in enumerate(model.model.layers):
            attention = layer.self_attn
            for target in targets:
                base = getattr(attention, target, None)
                if base is None:
                    raise RuntimeError(
                        f"Qwen MLX adapter target is missing: layer {layer_index} {target}"
                    )
                prefix = f"base_model.model.layers.{layer_index}.self_attn.{target}"
                key_a = prefix + ".lora_A.weight"
                key_b = prefix + ".lora_B.weight"
                if key_a not in adapter_weights or key_b not in adapter_weights:
                    raise RuntimeError(f"Qwen MLX adapter weights are missing for {prefix}")
                lora = LoRALinear.from_base(
                    base,
                    r=rank,
                    dropout=0.0,
                    scale=scale,
                )
                # PEFT stores A=(r,in), B=(out,r), while MLX expects
                # lora_a=(in,r), lora_b=(r,out).
                lora.lora_a = adapter_weights[key_a].T
                lora.lora_b = adapter_weights[key_b].T
                setattr(attention, target, lora)
                used_keys.update((key_a, key_b))

        unexpected = set(adapter_weights) - used_keys
        if unexpected:
            raise RuntimeError(
                f"Qwen MLX adapter contains {len(unexpected)} unsupported tensors"
            )
    inputs = tokenizer(
        [text],
        padding=True,
        truncation=True,
        max_length=int(getattr(pipeline, "max_length", 512)),
        return_tensors="mlx",
    )
    outputs = model(inputs["input_ids"], attention_mask=inputs["attention_mask"])
    mx.eval(outputs.text_embeds)
    return np.asarray(outputs.text_embeds, dtype=np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    args = parser.parse_args()

    started = time.perf_counter()
    payload = json.load(sys.stdin)
    bundle = joblib.load(args.artifact)
    pipeline = bundle["pipeline"]
    runtime = os.getenv("QWEN_RUNTIME", "torch").strip().lower() or "torch"
    if runtime == "mlx":
        embeddings = _mlx_embeddings(pipeline, payload)
        probabilities = np.asarray(
            pipeline.predict_proba_from_embeddings([payload], embeddings)[0],
            dtype=float,
        )
    elif runtime == "torch":
        probabilities = np.asarray(pipeline.predict_proba([payload])[0], dtype=float)
    else:
        raise RuntimeError(f"Unsupported QWEN_RUNTIME: {runtime}")
    order = np.argsort(probabilities)[::-1][:3]
    classes = [str(value) for value in pipeline.classes_]
    alternatives = [
        {"label": classes[int(index)], "confidence": round(float(probabilities[int(index)]), 4)}
        for index in order
    ]
    metadata = bundle.get("metadata", {})
    result = {
        "category": {
            "label": alternatives[0]["label"],
            "confidence": alternatives[0]["confidence"],
            "alternatives": alternatives,
        },
        "model_provenance": {
            "candidate_id": metadata.get("version", "v5.2-qwen4b-lite"),
            "model_id": metadata.get("model_id", "Qwen/Qwen3-Embedding-4B"),
            "model_revision": metadata.get("model_revision"),
            "profile": "qwen_recheck",
            "embedding_dim": metadata.get("embedding_dim"),
            "runtime": "mlx-4bit" if runtime == "mlx" else "torch-mps",
            "ephemeral": True,
        },
        "latency_seconds": round(time.perf_counter() - started, 3),
    }
    sys.stdout.write(json.dumps(result, ensure_ascii=False) + "\n")
    sys.stdout.flush()
    # MLX/PyTorch can spend a long time tearing down accelerator state during
    # normal interpreter shutdown. This worker owns no persistent resources;
    # exiting the process immediately releases all model memory to macOS.
    os._exit(0)


if __name__ == "__main__":
    main()
