from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np


def embedding_cache_key(model_id: str, revision: str, texts: Sequence[str], *, prefix: str = "") -> str:
    payload = {
        "schema": 1,
        "model_id": model_id,
        "revision": revision,
        "prefix": prefix,
        "texts": [str(text) for text in texts],
    }
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _normalized(matrix: np.ndarray) -> np.ndarray:
    values = np.asarray(matrix, dtype=np.float32)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(norms, 1e-12)


def load_or_encode_embeddings(
    encoder: Any,
    texts: Sequence[str],
    *,
    model_id: str,
    revision: str,
    cache_dir: Path,
    prefix: str = "",
    batch_size: int = 16,
) -> tuple[np.ndarray, dict[str, Any]]:
    key = embedding_cache_key(model_id, revision, texts, prefix=prefix)
    cache_dir.mkdir(parents=True, exist_ok=True)
    matrix_path = cache_dir / f"{key}.npy"
    metadata_path = cache_dir / f"{key}.json"
    if matrix_path.exists() and metadata_path.exists():
        return np.load(matrix_path, allow_pickle=False), json.loads(metadata_path.read_text(encoding="utf-8"))
    if encoder is None:
        raise FileNotFoundError(f"Embedding cache miss for {model_id}@{revision}: {key}")
    prefixed = [f"{prefix}{text}" for text in texts]
    started = time.perf_counter()
    matrix = encoder.encode(
        prefixed,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    matrix = _normalized(np.asarray(matrix, dtype=np.float32))
    encode_seconds = time.perf_counter() - started
    metadata = {
        "schema": 1,
        "cache_key": key,
        "model_id": model_id,
        "revision": revision,
        "prefix": prefix,
        "count": len(texts),
        "shape": list(matrix.shape),
        "dtype": str(matrix.dtype),
        "normalized": True,
        "encode_seconds": round(encode_seconds, 4),
        "encode_ms_per_text": round(1_000 * encode_seconds / max(1, len(texts)), 4),
    }
    np.save(matrix_path, matrix, allow_pickle=False)
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return matrix, metadata


class DenseRetriever:
    def __init__(self, document_embeddings: np.ndarray, *, query_embeddings: np.ndarray | None = None):
        self.document_embeddings = _normalized(document_embeddings)
        self.query_embeddings = _normalized(query_embeddings) if query_embeddings is not None else self.document_embeddings

    def score_index(self, query_index: int) -> np.ndarray:
        return np.asarray(self.query_embeddings[query_index] @ self.document_embeddings.T, dtype=float)
