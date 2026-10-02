from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from scipy import sparse


class LazyEmbeddingClassifier:
    """Deployment head that loads its pinned encoder once, on first inference."""

    def __init__(self, artifact: Mapping[str, Any], *, cache_dir: Path | None = None):
        self.model_id = str(artifact["model_id"])
        self.encoder_path = str(artifact.get("encoder_path") or "")
        self.revision = str(artifact["revision"])
        self.metadata_fields = tuple(artifact["metadata_fields"])
        self.metadata_scale = float(artifact["metadata_scale"])
        self.metadata_encoder = artifact["metadata_encoder"]
        self.classifier = artifact["classifier"]
        self.instruction = artifact.get("instruction")
        self.cache_dir = cache_dir
        self.classes_ = self.classifier.classes_
        self._encoder: Any = None

    def _load_encoder(self) -> Any:
        if self._encoder is None:
            import torch
            from sentence_transformers import SentenceTransformer

            project_root = Path(__file__).resolve().parents[4]
            local_encoder = project_root / self.encoder_path if self.encoder_path else None
            source = str(local_encoder) if local_encoder and local_encoder.exists() else self.model_id
            self._encoder = SentenceTransformer(
                source,
                revision=None if source != self.model_id else self.revision,
                cache_folder=str(self.cache_dir) if self.cache_dir else None,
                trust_remote_code=False,
                model_kwargs={"torch_dtype": torch.bfloat16 if torch.cuda.is_available() else torch.float32},
            )
        return self._encoder

    def predict_proba(self, values: Sequence[Mapping[str, Any]]) -> np.ndarray:
        encoder = self._load_encoder()
        kwargs: dict[str, Any] = {
            "normalize_embeddings": True,
            "convert_to_numpy": True,
            "show_progress_bar": False,
        }
        if self.instruction:
            kwargs["prompt"] = f"Instruct: {self.instruction}\nQuery:"
        embeddings = np.asarray(
            encoder.encode([str(value.get("description") or "") for value in values], **kwargs),
            dtype=np.float32,
        )
        matrix: Any = sparse.csr_matrix(embeddings)
        if self.metadata_scale:
            metadata = np.asarray(
                [[str(value.get(field) or "") for field in self.metadata_fields] for value in values],
                dtype=object,
            )
            encoded = self.metadata_encoder.transform(metadata) * self.metadata_scale
            matrix = sparse.hstack([matrix, encoded], format="csr")
        return np.asarray(self.classifier.predict_proba(matrix))

    def predict(self, values: Sequence[Mapping[str, Any]]) -> np.ndarray:
        probabilities = self.predict_proba(values)
        return np.asarray(self.classes_)[probabilities.argmax(axis=1)]
