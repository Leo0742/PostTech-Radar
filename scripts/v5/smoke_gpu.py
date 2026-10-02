from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
DEFAULT_OUTPUT = PROJECT_ROOT / "artifacts" / "gpu_research_v5" / "environment" / "gpu_smoke.json"


def _resolve(model_id: str, revision: str | None) -> tuple[str, Any]:
    if revision:
        return revision, None
    from huggingface_hub import model_info

    info = model_info(model_id)
    resolved = str(getattr(info, "sha", "") or "")
    if not resolved:
        raise RuntimeError(f"No immutable Hugging Face revision returned for {model_id}")
    card_data = getattr(info, "card_data", None)
    license_value = getattr(card_data, "license", None) if card_data is not None else None
    return resolved, license_value


def main() -> None:
    parser = argparse.ArgumentParser(description="Load one embedding model on CUDA and run a minimal V5 smoke test")
    parser.add_argument("--model-id", default="Qwen/Qwen3-Embedding-8B")
    parser.add_argument("--revision")
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--hash-model-snapshot", action="store_true")
    args = parser.parse_args()

    import numpy as np
    import torch
    from sentence_transformers import SentenceTransformer

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available")

    revision, license_value = _resolve(args.model_id, args.revision)
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    model = SentenceTransformer(
        args.model_id,
        revision=revision,
        trust_remote_code=True,
        device="cuda",
        model_kwargs={"torch_dtype": torch.float16},
    )
    model.max_seq_length = args.max_length
    texts = [
        "Не работает QR-код в личном кабинете",
        "Не приходит push уведомление по отправлению",
        "Не удается отследить почтовое отправление",
    ]
    embeddings = np.asarray(
        model.encode(
            texts,
            batch_size=max(1, args.batch_size),
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ),
        dtype=float,
    )
    elapsed = time.perf_counter() - started
    norms = np.linalg.norm(embeddings, axis=1)
    payload: dict[str, Any] = {
        "status": "ok",
        "model_id": args.model_id,
        "revision": revision,
        "license": license_value,
        "rows": int(embeddings.shape[0]),
        "embedding_dimension": int(embeddings.shape[1]),
        "max_length": args.max_length,
        "batch_size": args.batch_size,
        "finite": bool(np.isfinite(embeddings).all()),
        "normalized": bool(np.allclose(norms, 1.0, atol=1e-3)),
        "seconds": round(elapsed, 6),
        "peak_vram_bytes": int(torch.cuda.max_memory_allocated()),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(0),
        "cosine_0_1": float(np.dot(embeddings[0], embeddings[1])) if embeddings.shape[0] >= 2 else math.nan,
    }
    if args.hash_model_snapshot:
        from huggingface_hub import snapshot_download

        from scripts.v5.capture_environment import hash_model_snapshot

        snapshot = Path(snapshot_download(args.model_id, revision=revision, local_files_only=True))
        payload["model_snapshot"] = hash_model_snapshot(snapshot)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
