from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from scripts.v5_experiment import INSTRUCTION_REGISTRY

MISSING = "__MISSING__"


def _sha256_payload(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class DiskEmbeddingCache:
    def __init__(self, root: Path, backend: Callable[[Sequence[str], Mapping[str, Any]], np.ndarray]) -> None:
        self.root = Path(root)
        self.backend = backend

    def __call__(self, texts: Sequence[str], candidate: Mapping[str, Any]) -> np.ndarray:
        key = _sha256_payload({
            "model_id": candidate.get("model_id"),
            "model_revision": candidate.get("model_revision"),
            "feature_mode": candidate.get("feature_mode"),
            "instruction": candidate.get("instruction"),
            "max_length": int(candidate.get("max_length", 0)),
            "texts": list(texts),
        })
        array_path = self.root / f"{key}.npy"
        if array_path.exists():
            return np.load(array_path, allow_pickle=False)

        matrix = np.asarray(self.backend(texts, candidate), dtype=np.float32)
        self.root.mkdir(parents=True, exist_ok=True)
        with array_path.open("wb") as stream:
            np.save(stream, matrix, allow_pickle=False)
        return matrix


class SentenceTransformerBackend:
    def __init__(self, *, batch_size: int = 8) -> None:
        self.batch_size = max(1, int(batch_size))
        self._model: Any = None
        self._model_key: tuple[str, str] | None = None

    def __call__(self, texts: Sequence[str], candidate: Mapping[str, Any]) -> np.ndarray:
        import torch
        from sentence_transformers import SentenceTransformer

        model_id = str(candidate["model_id"])
        revision = str(candidate["model_revision"])
        key = (model_id, revision)
        if self._model is None or self._model_key != key:
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA GPU is required for the embedding runner")
            self._model = SentenceTransformer(
                model_id,
                revision=revision,
                trust_remote_code=True,
                device="cuda",
                model_kwargs={"torch_dtype": torch.float16},
            )
            self._model_key = key

        self._model.max_seq_length = int(candidate["max_length"])
        return np.asarray(
            self._model.encode(
                list(texts),
                batch_size=self.batch_size,
                show_progress_bar=True,
                convert_to_numpy=True,
                normalize_embeddings=True,
            ),
            dtype=np.float32,
        )


def _clean_structured(value: Any) -> str:
    if value is None:
        return MISSING
    text = " ".join(str(value).split())
    return text or MISSING


def build_structured_features(rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
    result: list[list[str]] = []
    for row in rows:
        registration = datetime.fromisoformat(str(row["registration_date"]).replace("Z", "+00:00"))
        result.append([
            _clean_structured(row.get("user")),
            _clean_structured(row.get("service")),
            _clean_structured(row.get("component")),
            _clean_structured(row.get("request_type")),
            _clean_structured(row.get("criticality")),
            _clean_structured(row.get("urgency")),
            _clean_structured(row.get("priority")),
            _clean_structured(row.get("service_class")),
            _clean_structured(row.get("timezone")),
            str(registration.month),
            str(registration.weekday()),
            str(registration.hour),
        ])
    return np.asarray(result, dtype=object)


def _apply_instruction(texts: Sequence[str], instruction_key: str) -> list[str]:
    instruction = INSTRUCTION_REGISTRY[instruction_key]
    if not instruction:
        return [str(text) for text in texts]
    return [f"Instruct: {instruction}\nQuery: {text}" for text in texts]


def _normalize_rows(values: np.ndarray) -> np.ndarray:
    matrix = np.asarray(values, dtype=float)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return np.divide(matrix, norms, out=np.zeros_like(matrix), where=norms > 0)
