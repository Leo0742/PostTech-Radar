from __future__ import annotations

import re
import time
from collections import defaultdict
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

_TOKEN_RE = re.compile(r"[a-zа-яё0-9]+(?:\.[0-9]+)?", re.IGNORECASE)


def tokenize(value: str) -> list[str]:
    return _TOKEN_RE.findall(value.lower())


def minmax_normalize(scores: np.ndarray) -> np.ndarray:
    values = np.asarray(scores, dtype=float)
    if not len(values):
        return values
    lower = float(values.min())
    upper = float(values.max())
    if upper <= lower:
        return np.zeros_like(values)
    return (values - lower) / (upper - lower)


def reciprocal_rank_fusion(rankings: Sequence[Sequence[int]], *, k: int = 60) -> dict[int, float]:
    scores: dict[int, float] = defaultdict(float)
    for ranking in rankings:
        for rank, identifier in enumerate(ranking, start=1):
            scores[int(identifier)] += 1.0 / (k + rank)
    return dict(sorted(scores.items(), key=lambda item: (-item[1], item[0])))


def duplicate_free_ranking(scores: np.ndarray, *, query_index: int, groups: Sequence[str], limit: int) -> list[int]:
    query_group = groups[query_index]
    order = np.argsort(-np.asarray(scores), kind="stable")
    return [int(index) for index in order if int(index) != query_index and groups[int(index)] != query_group][:limit]


def learn_relevance_threshold(
    positive_scores: Sequence[float],
    negative_scores: Sequence[float],
    *,
    target_precision: float = 0.9,
) -> float:
    values = [(float(score), True) for score in positive_scores] + [(float(score), False) for score in negative_scores]
    if not positive_scores:
        raise ValueError("positive relevance scores are required")
    candidates = sorted({score for score, _ in values})
    valid = []
    for threshold in candidates:
        accepted = [relevant for score, relevant in values if score >= threshold]
        if not accepted:
            continue
        precision = sum(accepted) / len(accepted)
        if precision >= target_precision:
            valid.append((len(accepted), -threshold, threshold))
    return round(float(max(valid)[2]), 8) if valid else round(max(positive_scores) + 1e-9, 8)


class LexicalRetriever:
    def __init__(self, kind: str):
        if kind not in {"word_tfidf", "char_tfidf", "word_char_tfidf", "bm25"}:
            raise ValueError(f"Unknown lexical retriever: {kind}")
        self.kind = kind
        self.texts: list[str] = []
        self.vectorizer: Any = None
        self.matrix: Any = None
        self.word_vectorizer: Any = None
        self.word_matrix: Any = None
        self.char_vectorizer: Any = None
        self.char_matrix: Any = None
        self.bm25: Any = None

    def fit(self, texts: Sequence[str]) -> LexicalRetriever:
        self.texts = [str(text) for text in texts]
        if self.kind == "bm25":
            from rank_bm25 import BM25Okapi

            self.bm25 = BM25Okapi([tokenize(text) for text in self.texts])
        elif self.kind == "word_char_tfidf":
            self.word_vectorizer = TfidfVectorizer(ngram_range=(1, 3), min_df=1, sublinear_tf=True, max_features=30_000)
            self.char_vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 6), min_df=1, sublinear_tf=True, max_features=40_000)
            self.word_matrix = self.word_vectorizer.fit_transform(self.texts)
            self.char_matrix = self.char_vectorizer.fit_transform(self.texts)
        else:
            options = (
                {"ngram_range": (1, 3), "max_features": 30_000}
                if self.kind == "word_tfidf"
                else {"analyzer": "char_wb", "ngram_range": (2, 6), "max_features": 40_000}
            )
            self.vectorizer = TfidfVectorizer(min_df=1, sublinear_tf=True, **options)
            self.matrix = self.vectorizer.fit_transform(self.texts)
        return self

    def score(self, query: str) -> np.ndarray:
        if self.kind == "bm25":
            return minmax_normalize(np.asarray(self.bm25.get_scores(tokenize(query)), dtype=float))
        if self.kind == "word_char_tfidf":
            word = (self.word_vectorizer.transform([query]) @ self.word_matrix.T).toarray()[0]
            char = (self.char_vectorizer.transform([query]) @ self.char_matrix.T).toarray()[0]
            return 0.5 * word + 0.5 * char
        return (self.vectorizer.transform([query]) @ self.matrix.T).toarray()[0]


def _evaluate_scores(
    score_provider: Callable[[int, dict[str, Any]], np.ndarray],
    rows: list[dict[str, Any]],
    *,
    model: str,
) -> dict[str, Any]:
    groups = [str(row.get("normalized_description") or f"request:{row['request_id']}") for row in rows]
    hit_counts = {1: 0, 3: 0, 5: 0}
    precision_sums = {1: 0.0, 3: 0.0, 5: 0.0}
    reciprocal_ranks: list[float] = []
    positive_scores: list[float] = []
    negative_scores: list[float] = []
    examples = []
    started = time.perf_counter()
    for query_index, row in enumerate(rows):
        scores = score_provider(query_index, row)
        ranking = duplicate_free_ranking(scores, query_index=query_index, groups=groups, limit=50)
        relevant = [position for position, candidate in enumerate(ranking, start=1) if rows[candidate]["category"] == row["category"]]
        reciprocal_ranks.append(1.0 / relevant[0] if relevant else 0.0)
        for k in (1, 3, 5):
            relevant_count = sum(rows[candidate]["category"] == row["category"] for candidate in ranking[:k])
            hit_counts[k] += int(relevant_count > 0)
            precision_sums[k] += relevant_count / k
        valid = [index for index in range(len(rows)) if index != query_index and groups[index] != groups[query_index]]
        same = [float(scores[index]) for index in valid if rows[index]["category"] == row["category"]]
        different = [float(scores[index]) for index in valid if rows[index]["category"] != row["category"]]
        if same:
            positive_scores.append(max(same))
        if different:
            negative_scores.append(max(different))
        if len(examples) < 20:
            examples.append(
                {
                    "query_request_id": row["request_id"],
                    "query_category": row["category"],
                    "neighbors": [
                        {
                            "request_id": rows[index]["request_id"],
                            "category": rows[index]["category"],
                            "relevance": round(float(scores[index]), 6),
                        }
                        for index in ranking[:5]
                    ],
                }
            )
    elapsed = time.perf_counter() - started
    query_count = len(rows)
    threshold = learn_relevance_threshold(positive_scores, negative_scores, target_precision=0.9)
    return {
        "model": model,
        "queries": query_count,
        "duplicate_groups_excluded": True,
        "hit_rate_at_1": round(hit_counts[1] / query_count, 6),
        "hit_rate_at_3": round(hit_counts[3] / query_count, 6),
        "hit_rate_at_5": round(hit_counts[5] / query_count, 6),
        "precision_at_1": round(precision_sums[1] / query_count, 6),
        "precision_at_3": round(precision_sums[3] / query_count, 6),
        "precision_at_5": round(precision_sums[5] / query_count, 6),
        "mrr": round(float(np.mean(reciprocal_ranks)), 6),
        "latency_ms_per_query": round(1_000 * elapsed / query_count, 4),
        "rejection_threshold": threshold,
        "threshold_proxy": "90% precision separating best same-category score from best different-category score",
        "examples": examples,
    }


def evaluate_retriever(retriever: LexicalRetriever, rows: list[dict[str, Any]]) -> dict[str, Any]:
    return _evaluate_scores(
        lambda _index, row: retriever.score(str(row["description"])),
        rows,
        model=retriever.kind,
    )


def evaluate_dense_retriever(retriever: Any, rows: list[dict[str, Any]], *, model: str) -> dict[str, Any]:
    return _evaluate_scores(
        lambda index, _row: retriever.score_index(index),
        rows,
        model=model,
    )


def evaluate_hybrid_retriever(
    lexical: LexicalRetriever,
    dense: Any,
    rows: list[dict[str, Any]],
    *,
    dense_weight: float = 0.5,
) -> dict[str, Any]:
    if not 0 <= dense_weight <= 1:
        raise ValueError("dense_weight must be between zero and one")

    def scores(index: int, row: dict[str, Any]) -> np.ndarray:
        lexical_scores = minmax_normalize(lexical.score(str(row["description"])))
        dense_scores = minmax_normalize(dense.score_index(index))
        return (1.0 - dense_weight) * lexical_scores + dense_weight * dense_scores

    result = _evaluate_scores(scores, rows, model=f"hybrid/{lexical.kind}")
    result["fusion"] = {
        "lexical_weight": round(1.0 - dense_weight, 6),
        "dense_weight": round(dense_weight, 6),
    }
    return result
