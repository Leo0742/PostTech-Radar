from __future__ import annotations

import numpy as np
from app.ml.v3.embeddings import DenseRetriever
from app.ml.v3.retrieval import (
    LexicalRetriever,
    duplicate_free_ranking,
    evaluate_dense_retriever,
    evaluate_hybrid_retriever,
    evaluate_retriever,
    learn_relevance_threshold,
    minmax_normalize,
    reciprocal_rank_fusion,
)


def test_duplicate_free_ranking_excludes_query_and_exact_description_group() -> None:
    scores = np.asarray([1.0, 0.99, 0.8, 0.7])
    groups = ["same", "same", "other", "fourth"]

    assert duplicate_free_ranking(scores, query_index=0, groups=groups, limit=3) == [2, 3]


def test_score_normalization_and_rrf_have_explicit_stable_semantics() -> None:
    assert minmax_normalize(np.asarray([4.0, 2.0, 0.0])).tolist() == [1.0, 0.5, 0.0]
    assert minmax_normalize(np.asarray([3.0, 3.0])).tolist() == [0.0, 0.0]

    fused = reciprocal_rank_fusion([[3, 1, 2], [1, 2, 4]], k=10)

    assert list(fused) == [1, 2, 3, 4]
    assert fused[1] > fused[2] > fused[3] > fused[4]


def test_bm25_retrieves_domain_tokens_and_threshold_rejects_low_scores() -> None:
    texts = ["не работает qr код", "ошибка импорта zip архива", "личный кабинет не открывается"]
    retriever = LexicalRetriever("bm25").fit(texts)

    scores = retriever.score("проблема qr кода")
    threshold = learn_relevance_threshold([0.9, 0.8, 0.7], [0.6, 0.2, 0.1], target_precision=1.0)

    assert int(scores.argmax()) == 0
    assert threshold == 0.7


def test_retrieval_evaluation_uses_all_queries_and_excludes_duplicate_groups() -> None:
    rows = [
        {"request_id": "1", "description": "qr код ошибка", "category": "A", "normalized_description": "qr код ошибка"},
        {"request_id": "2", "description": "qr код подключение", "category": "A", "normalized_description": "qr код подключение"},
        {"request_id": "3", "description": "zip архив ошибка", "category": "B", "normalized_description": "zip архив ошибка"},
        {"request_id": "4", "description": "zip архив импорт", "category": "B", "normalized_description": "zip архив импорт"},
    ]
    retriever = LexicalRetriever("word_tfidf").fit([row["description"] for row in rows])

    result = evaluate_retriever(retriever, rows)

    assert result["queries"] == 4
    assert result["duplicate_groups_excluded"] is True
    assert result["hit_rate_at_1"] == 1.0
    assert result["mrr"] == 1.0


def test_dense_and_hybrid_evaluation_share_duplicate_free_protocol() -> None:
    rows = [
        {"request_id": "1", "description": "qr код ошибка", "category": "A", "normalized_description": "one"},
        {"request_id": "2", "description": "qr код вход", "category": "A", "normalized_description": "two"},
        {"request_id": "3", "description": "zip импорт", "category": "B", "normalized_description": "three"},
        {"request_id": "4", "description": "zip архив", "category": "B", "normalized_description": "four"},
    ]
    embeddings = np.asarray([[1, 0], [0.9, 0.1], [0, 1], [0.1, 0.9]], dtype=np.float32)
    dense = DenseRetriever(embeddings)
    lexical = LexicalRetriever("word_tfidf").fit([row["description"] for row in rows])

    dense_result = evaluate_dense_retriever(dense, rows, model="fake-dense")
    hybrid_result = evaluate_hybrid_retriever(lexical, dense, rows, dense_weight=0.6)

    assert dense_result["mrr"] == 1.0
    assert hybrid_result["mrr"] == 1.0
    assert hybrid_result["fusion"] == {"lexical_weight": 0.4, "dense_weight": 0.6}
