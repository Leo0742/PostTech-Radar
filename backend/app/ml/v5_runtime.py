from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

MISSING = "__MISSING__"
SPECIALIST_FIELDS = (
    "service",
    "component",
    "request_type",
    "user",
    "criticality",
    "urgency",
    "priority",
)


def _clean(value: Any) -> str:
    if value is None:
        return MISSING
    text = " ".join(str(value).split())
    return text or MISSING


def build_v5_structured_features(rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
    """Build the V5 metadata branch and tolerate missing demo-time fields."""
    result: list[list[str]] = []
    for row in rows:
        registration = str(row.get("registration_date") or "").strip()
        month = weekday = hour = MISSING
        if registration:
            try:
                moment = datetime.fromisoformat(registration.replace("Z", "+00:00"))
                month, weekday, hour = str(moment.month), str(moment.weekday()), str(moment.hour)
            except ValueError:
                pass
        result.append([
            _clean(row.get("user")),
            _clean(row.get("service")),
            _clean(row.get("component")),
            _clean(row.get("request_type")),
            _clean(row.get("criticality")),
            _clean(row.get("urgency")),
            _clean(row.get("priority")),
            _clean(row.get("service_class")),
            _clean(row.get("timezone")),
            month,
            weekday,
            hour,
        ])
    return np.asarray(result, dtype=object)


def specialist_text(row: Mapping[str, Any]) -> str:
    description = " ".join(str(row.get("description") or "").split())
    metadata = " | ".join(str(row.get(field) or "") for field in SPECIALIST_FIELDS)
    return f"{description} META {metadata}"


def _normalize(values: np.ndarray) -> np.ndarray:
    matrix = np.asarray(values, dtype=float)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return np.divide(matrix, norms, out=np.zeros_like(matrix), where=norms > 0)


def _softmax(values: np.ndarray, temperature: float) -> np.ndarray:
    scaled = np.asarray(values, dtype=float) / max(float(temperature), 1e-6)
    scaled -= np.max(scaled, axis=1, keepdims=True)
    exp = np.exp(scaled)
    return exp / exp.sum(axis=1, keepdims=True)


def _aligned_probabilities(model: Any, values: np.ndarray, labels: Sequence[str]) -> np.ndarray:
    local = np.asarray(model.predict_proba(values), dtype=float)
    positions = {str(label): index for index, label in enumerate(labels)}
    result = np.zeros((len(values), len(labels)), dtype=float)
    for source, label in enumerate(model.classes_):
        target = positions.get(str(label))
        if target is not None:
            result[:, target] = local[:, source]
    sums = result.sum(axis=1, keepdims=True)
    return np.divide(result, sums, out=np.zeros_like(result), where=sums > 0)


class DecisionSoftmaxClassifier:
    """Expose LinearSVC-style decision scores through a predict_proba contract.

    The customer-adapted Qwen head is selected with leakage-safe OOF validation
    as a LinearSVC.  QwenV5CategoryPipeline expects a probability-like interface
    for blending with prototype and kNN heads, so this wrapper applies one fixed
    softmax temperature without changing class ordering.
    """

    def __init__(self, estimator: Any, temperature: float = 1.0) -> None:
        self.estimator = estimator
        self.temperature = float(temperature)

    @property
    def classes_(self) -> np.ndarray:
        return np.asarray(self.estimator.classes_, dtype=object)

    def predict_proba(self, values: Any) -> np.ndarray:
        scores = np.asarray(self.estimator.decision_function(values), dtype=float)
        if scores.ndim == 1:
            scores = np.column_stack([-scores, scores])
        return _softmax(scores, self.temperature)

    def predict(self, values: Any) -> np.ndarray:
        probabilities = self.predict_proba(values)
        return self.classes_[probabilities.argmax(axis=1)]


class QwenV5CategoryPipeline:
    """Serializable V5 inference pipeline with lazy loading of the pinned Qwen encoder."""

    def __init__(
        self,
        *,
        labels: Sequence[str],
        model_id: str,
        model_revision: str,
        instruction: str,
        max_length: int,
        embedding_dim: int,
        metadata_encoder: Any,
        supervised_model: Any,
        prototype_centroids: np.ndarray,
        knn_model: Any,
        specialist_vectorizer: Any | None = None,
        specialist_model: Any | None = None,
        specialist_pairs: Sequence[Sequence[str]] = (),
        specialist_threshold: float = 0.15,
        metadata_scale: float = 1.0,
        prototype_temperature: float = 0.08,
        batch_size: int = 8,
        blend_weights: Sequence[float] = (0.90, 0.05, 0.05),
        adapter_dir: str = "",
    ) -> None:
        self.labels = [str(value) for value in labels]
        self.model_id = model_id
        self.model_revision = model_revision
        self.instruction = instruction
        self.max_length = int(max_length)
        self.embedding_dim = int(embedding_dim)
        self.metadata_encoder = metadata_encoder
        self.supervised_model = supervised_model
        self.prototype_centroids = _normalize(np.asarray(prototype_centroids, dtype=float))
        self.knn_model = knn_model
        self.specialist_vectorizer = specialist_vectorizer
        self.specialist_model = specialist_model
        self.specialist_pairs = {frozenset(map(str, pair)) for pair in specialist_pairs}
        self.specialist_threshold = float(specialist_threshold)
        self.metadata_scale = float(metadata_scale)
        self.prototype_temperature = float(prototype_temperature)
        self.batch_size = int(batch_size)
        self.blend_weights = tuple(float(value) for value in blend_weights)
        self.adapter_dir = str(adapter_dir)
        self._sentence_model: Any | None = None

    @property
    def classes_(self) -> np.ndarray:
        return np.asarray(self.labels, dtype=object)

    def __getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        state["_sentence_model"] = None
        return state

    def _load_sentence_model(self) -> Any:
        if self._sentence_model is not None:
            return self._sentence_model
        import torch
        from sentence_transformers import SentenceTransformer

        device = resolve_qwen_device(torch, os.getenv("QWEN_DEVICE", "auto"))
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("V5 Qwen embedding runtime requires an available CUDA GPU")
        if device == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("V5 Qwen embedding runtime requires Apple MPS support")
        local_model_dir = os.getenv("QWEN_LOCAL_MODEL_DIR", "").strip()
        model_name_or_path = local_model_dir or self.model_id
        revision = None if local_model_dir else self.model_revision
        torch_dtype = torch.float16 if device in {"cuda", "mps"} else torch.float32
        self._sentence_model = SentenceTransformer(
            model_name_or_path,
            revision=revision,
            trust_remote_code=True,
            device=device,
            model_kwargs={"torch_dtype": torch_dtype},
        )
        adapter_dir = str(getattr(self, "adapter_dir", "") or "").strip()
        if adapter_dir:
            from peft import PeftModel

            adapter_path = Path(adapter_dir)
            if not adapter_path.is_absolute():
                adapter_path = Path(__file__).resolve().parents[3] / adapter_path
            if not adapter_path.exists():
                raise RuntimeError(f"Qwen PEFT adapter is missing: {adapter_path}")
            self._sentence_model[0].auto_model = PeftModel.from_pretrained(
                self._sentence_model[0].auto_model,
                str(adapter_path),
                is_trainable=False,
            )
        self._sentence_model.max_seq_length = self.max_length
        return self._sentence_model

    def _embed(self, rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
        texts = [" ".join(str(row.get("description") or "").split()) for row in rows]
        if self.instruction:
            texts = [f"Instruct: {self.instruction}\nQuery: {text}" for text in texts]
        values = self._load_sentence_model().encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return _normalize(np.asarray(values, dtype=float)[:, : self.embedding_dim])

    def _apply_specialist(self, rows: Sequence[Mapping[str, Any]], probabilities: np.ndarray) -> np.ndarray:
        if self.specialist_vectorizer is None or self.specialist_model is None or not self.specialist_pairs:
            return probabilities
        matrix = self.specialist_vectorizer.transform([specialist_text(row) for row in rows])
        decision = np.asarray(self.specialist_model.decision_function(matrix), dtype=float)
        classes = [str(value) for value in self.specialist_model.classes_]
        class_index = {label: index for index, label in enumerate(classes)}
        result = np.asarray(probabilities, dtype=float).copy()
        for row_index in range(len(rows)):
            order = np.argsort(-result[row_index])
            first = self.labels[int(order[0])]
            second = self.labels[int(order[1])]
            if frozenset((first, second)) not in self.specialist_pairs:
                continue
            if first not in class_index or second not in class_index:
                continue
            margin = decision[row_index, class_index[second]] - decision[row_index, class_index[first]]
            if margin > self.specialist_threshold:
                left, right = int(order[0]), int(order[1])
                result[row_index, left], result[row_index, right] = (
                    result[row_index, right],
                    result[row_index, left],
                )
        return result

    def predict_proba(self, values: Sequence[Any]) -> np.ndarray:
        rows: list[dict[str, Any]] = []
        for value in values:
            if isinstance(value, Mapping):
                rows.append(dict(value))
            else:
                rows.append({"description": str(value)})
        embeddings = self._embed(rows)
        return self.predict_proba_from_embeddings(rows, embeddings)

    def predict_proba_from_embeddings(
        self,
        values: Sequence[Any],
        embeddings: np.ndarray,
    ) -> np.ndarray:
        """Run the trained V5 heads on externally produced Qwen embeddings.

        The operator recheck path uses this to run the same frozen classifier
        with a memory-efficient MLX encoder on Apple Silicon. The embedding
        runtime changes, while the metadata branch, classifier heads and
        specialist logic remain identical to the deployment artifact.
        """
        rows: list[dict[str, Any]] = []
        for value in values:
            if isinstance(value, Mapping):
                rows.append(dict(value))
            else:
                rows.append({"description": str(value)})
        embeddings = _normalize(np.asarray(embeddings, dtype=float)[:, : self.embedding_dim])
        if len(embeddings) != len(rows):
            raise ValueError("Embedding row count must match input row count")
        metadata = np.asarray(
            self.metadata_encoder.transform(build_v5_structured_features(rows)),
            dtype=float,
        )
        metadata *= float(getattr(self, "metadata_scale", 1.0))
        supervised_x = np.concatenate([embeddings, metadata], axis=1)
        supervised = _aligned_probabilities(self.supervised_model, supervised_x, self.labels)
        prototype = _softmax(
            embeddings @ self.prototype_centroids.T,
            temperature=float(getattr(self, "prototype_temperature", 0.08)),
        )
        knn = _aligned_probabilities(self.knn_model, embeddings, self.labels)
        blend = tuple(getattr(self, "blend_weights", (0.90, 0.05, 0.05)))
        if len(blend) != 3:
            raise ValueError("Qwen blend_weights must contain supervised, prototype and kNN weights")
        probabilities = blend[0] * supervised + blend[1] * prototype + blend[2] * knn
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        return self._apply_specialist(rows, probabilities)

    def predict(self, values: Sequence[Any]) -> np.ndarray:
        probabilities = self.predict_proba(values)
        return self.classes_[probabilities.argmax(axis=1)]
def resolve_qwen_device(torch_module: Any, requested: str) -> str:
    requested = requested.strip().lower() or "auto"
    if requested == "auto":
        if torch_module.cuda.is_available():
            return "cuda"
        if torch_module.backends.mps.is_available():
            return "mps"
        return "cpu"
    if requested not in {"cuda", "mps", "cpu"}:
        raise ValueError("QWEN_DEVICE must be one of: auto, cuda, mps, cpu")
    return requested
