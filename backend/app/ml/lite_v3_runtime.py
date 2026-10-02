from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from scipy import sparse

from app.ml.lite_v2_runtime import LiteV2CategoryPipeline, _softmax


def _normalize(values: np.ndarray) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return np.divide(matrix, norms, out=np.zeros_like(matrix), where=norms > 0)


class LiteV3DistilledCategoryPipeline(LiteV2CategoryPipeline):
    """Fast local category model distilled from saved Qwen4B embeddings.

    Runtime inference uses multilingual MiniLM plus a train-only Ridge projection
    learned from the saved Qwen4B teacher embeddings. Qwen4B itself is not loaded
    by this pipeline.
    """

    def __init__(
        self,
        *,
        projector: Any,
        minilm_model_id: str,
        local_model_dir: str,
        max_seq_length: int = 256,
        batch_size: int = 16,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.projector = projector
        self.minilm_model_id = str(minilm_model_id)
        self.local_model_dir = str(local_model_dir)
        self.max_seq_length = int(max_seq_length)
        self.batch_size = int(batch_size)
        self._sentence_model: Any | None = None

    def __getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        state["_sentence_model"] = None
        return state

    def _load_sentence_model(self) -> Any:
        if self._sentence_model is not None:
            return self._sentence_model

        import torch
        from sentence_transformers import SentenceTransformer

        requested = (os.getenv("LITE_MINILM_DEVICE", "auto").strip().lower() or "auto")
        if requested == "auto":
            if torch.backends.mps.is_available():
                device = "mps"
            elif torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"
        elif requested in {"mps", "cuda", "cpu"}:
            device = requested
        else:
            raise ValueError("LITE_MINILM_DEVICE must be one of: auto, mps, cuda, cpu")

        override = os.getenv("LITE_MINILM_MODEL_DIR", "").strip()
        if override:
            model_path = Path(override)
        else:
            root = Path(__file__).resolve().parents[3]
            model_path = root / self.local_model_dir
        if not model_path.exists():
            raise RuntimeError(f"Lite V3 MiniLM model is missing: {model_path}")

        self._sentence_model = SentenceTransformer(
            str(model_path),
            device=device,
            local_files_only=True,
        )
        self._sentence_model.max_seq_length = self.max_seq_length
        return self._sentence_model

    def _projected_embeddings(self, rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
        texts = [" ".join(str(row.get("description") or "").split()) for row in rows]
        student = self._load_sentence_model().encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        projected = self.projector.predict(np.asarray(student, dtype=np.float32))
        return _normalize(projected)

    def predict_proba(self, rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
        rows = [dict(row) for row in rows]
        text_values = self.vectorizer.transform(self._texts(rows))
        metadata_values = self._metadata(rows)
        projected = sparse.csr_matrix(self._projected_embeddings(rows))
        values = sparse.hstack(
            [text_values, metadata_values * self.metadata_scale, projected],
            format="csr",
        )
        local_probabilities = _softmax(self.classifier.decision_function(values), self.temperature)

        result = np.zeros((len(rows), len(self.labels)), dtype=float)
        positions = {label: index for index, label in enumerate(self.labels)}
        for source, label in enumerate(self.classifier.classes_):
            target = positions.get(str(label))
            if target is not None:
                result[:, target] = local_probabilities[:, source]
        sums = result.sum(axis=1, keepdims=True)
        normalized = np.divide(result, sums, out=np.zeros_like(result), where=sums > 0)
        return self._apply_specialist(rows, normalized)
