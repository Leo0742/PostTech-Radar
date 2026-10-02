from __future__ import annotations

import json

import numpy as np
from app.ml.v3.embeddings import DenseRetriever, embedding_cache_key, load_or_encode_embeddings
from app.ml.v3.experiments import evaluate_embedding_candidate
from app.ml.v3.protocol import build_v3_manifest


class FakeEncoder:
    def encode(self, texts, **kwargs):
        values = [[len(text), text.count("а") + 1] for text in texts]
        return np.asarray(values, dtype=np.float32)


def test_embedding_cache_key_includes_model_revision_and_corpus() -> None:
    base = embedding_cache_key("model", "rev-a", ["один", "два"], prefix="query: ")
    assert base != embedding_cache_key("model", "rev-b", ["один", "два"], prefix="query: ")
    assert base != embedding_cache_key("model", "rev-a", ["один", "три"], prefix="query: ")
    assert base != embedding_cache_key("model", "rev-a", ["один", "два"], prefix="")


def test_load_or_encode_embeddings_reuses_immutable_cache(tmp_path) -> None:
    texts = ["первая", "вторая"]
    first, metadata = load_or_encode_embeddings(
        FakeEncoder(), texts, model_id="fake", revision="abc", cache_dir=tmp_path, prefix="passage: "
    )
    second, second_metadata = load_or_encode_embeddings(
        None, texts, model_id="fake", revision="abc", cache_dir=tmp_path, prefix="passage: "
    )
    assert np.allclose(first, second)
    assert metadata == second_metadata
    assert metadata["shape"] == [2, 2]
    saved = json.loads(next(tmp_path.glob("*.json")).read_text(encoding="utf-8"))
    assert saved["revision"] == "abc"


def test_dense_retriever_returns_cosine_scores_for_precomputed_queries() -> None:
    matrix = np.asarray([[1.0, 0.0], [0.8, 0.2], [0.0, 1.0]], dtype=np.float32)
    retriever = DenseRetriever(matrix, query_embeddings=matrix)
    scores = retriever.score_index(0)
    assert scores[0] == 1.0
    assert scores[1] > scores[2]


def test_embedding_candidate_uses_frozen_group_protocol_and_safe_metadata_only() -> None:
    rows = []
    vectors = []
    for label_index, label in enumerate(("A", "B", "C")):
        for example in range(10):
            rows.append(
                {
                    "request_id": f"{label}-{example}",
                    "description": f"описание {label} {example}",
                    "normalized_description": f"описание {label} {example}",
                    "category": label,
                    "service": f"service-{example % 2}",
                }
            )
            vector = np.zeros(3, dtype=np.float32)
            vector[label_index] = 1.0
            vectors.append(vector)
    manifest = build_v3_manifest(rows, repeats=1, n_splits=2)
    result = evaluate_embedding_candidate(
        candidate_id="category/fake/v3",
        model_id="fake",
        revision="abc",
        embeddings=np.asarray(vectors),
        rows=rows,
        manifest=manifest,
        with_metadata=True,
    )
    assert result["sealed_holdout_accessed"] is False
    assert result["metrics"]["cv_macro_f1_mean"] == 1.0
