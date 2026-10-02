from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

from app.core.config import DATABASE_PATH  # noqa: E402
from app.ml.v3.embeddings import DenseRetriever  # noqa: E402
from app.ml.v3.retrieval import LexicalRetriever, duplicate_free_ranking, minmax_normalize  # noqa: E402
from app.services.data_service import load_rows  # noqa: E402

MODEL_ID = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
REVISION = "1427fd652930e4ba29e8149678df786c240d8825"


def _sample_indices(rows: list[dict], *, limit: int = 300, seed: int = 20260916) -> list[int]:
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        grouped[str(row["category"])].append(index)
    rng = np.random.default_rng(seed)
    output: list[int] = []
    per_group = max(1, limit // len(grouped))
    for label in sorted(grouped):
        values = np.asarray(grouped[label])
        rng.shuffle(values)
        output.extend(int(value) for value in values[:per_group])
    if len(output) < limit:
        remaining = np.asarray(sorted(set(range(len(rows))) - set(output)))
        rng.shuffle(remaining)
        output.extend(int(value) for value in remaining[: limit - len(output)])
    return sorted(output[:limit])


def _metrics(rankings: list[list[int]], queries: list[int], rows: list[dict]) -> dict:
    hits = {1: 0, 3: 0, 5: 0}
    reciprocal = []
    for query_index, ranking in zip(queries, rankings, strict=True):
        relevant = [position for position, index in enumerate(ranking, start=1) if rows[index]["category"] == rows[query_index]["category"]]
        reciprocal.append(1.0 / relevant[0] if relevant else 0.0)
        for k in hits:
            hits[k] += int(any(rows[index]["category"] == rows[query_index]["category"] for index in ranking[:k]))
    return {
        "queries": len(queries),
        "hit_rate_at_1": round(hits[1] / len(queries), 6),
        "hit_rate_at_3": round(hits[3] / len(queries), 6),
        "hit_rate_at_5": round(hits[5] / len(queries), 6),
        "mrr": round(float(np.mean(reciprocal)), 6),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark a pinned cross-encoder on hybrid top-k candidates")
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--revision", default=REVISION)
    parser.add_argument("--query-limit", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "artifacts/research/v3/retrieval/reranker.json")
    args = parser.parse_args()
    model_cache = PROJECT_ROOT / "work/model-cache-v3/huggingface"
    os.environ.setdefault("HF_HOME", str(model_cache))
    os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", str(model_cache))
    from sentence_transformers import CrossEncoder

    rows = [row for row in load_rows(DATABASE_PATH) if str(row.get("description") or "").strip()]
    texts = [str(row["description"]) for row in rows]
    groups = [str(row.get("normalized_description") or row["request_id"]) for row in rows]
    gte = json.loads(
        (PROJECT_ROOT / "artifacts/research/v3/embeddings/gte-multilingual-base.json").read_text(encoding="utf-8")
    )
    embedding_root = PROJECT_ROOT / "artifacts/research/v3/embedding-cache"
    document = np.load(embedding_root / f"{gte['encoding']['document']['cache_key']}.npy", allow_pickle=False)
    query = np.load(embedding_root / f"{gte['encoding']['query']['cache_key']}.npy", allow_pickle=False)
    dense = DenseRetriever(document, query_embeddings=query)
    lexical = LexicalRetriever("word_char_tfidf").fit(texts)
    queries = _sample_indices(rows, limit=args.query_limit)
    base_rankings = []
    for query_index in queries:
        scores = 0.5 * minmax_normalize(lexical.score(texts[query_index])) + 0.5 * minmax_normalize(dense.score_index(query_index))
        base_rankings.append(duplicate_free_ranking(scores, query_index=query_index, groups=groups, limit=20))

    load_started = time.perf_counter()
    reranker = CrossEncoder(
        args.model_id,
        revision=args.revision,
        cache_dir=str(model_cache),
        device="cpu",
        max_length=128,
    )
    load_seconds = time.perf_counter() - load_started
    pairs = [
        (texts[query_index], texts[candidate])
        for query_index, ranking in zip(queries, base_rankings, strict=True)
        for candidate in ranking
    ]
    started = time.perf_counter()
    scores = np.asarray(reranker.predict(pairs, batch_size=args.batch_size, show_progress_bar=True)).reshape(len(queries), 20)
    predict_seconds = time.perf_counter() - started
    reranked = [
        [ranking[position] for position in np.argsort(-query_scores, kind="stable")]
        for ranking, query_scores in zip(base_rankings, scores, strict=True)
    ]
    payload = {
        "status": "MEASURED_STRATIFIED_SAMPLE",
        "model_id": args.model_id,
        "revision": args.revision,
        "license": "Apache-2.0 (verify for overridden model)",
        "protocol": {
            "queries": len(queries),
            "seed": 20260916,
            "first_stage": "0.5 * normalized word_char_tfidf + 0.5 * normalized gte-multilingual-base",
            "candidates_per_query": 20,
            "duplicate_groups_excluded": True,
            "relevance_proxy": "same category",
        },
        "baseline": _metrics(base_rankings, queries, rows),
        "reranked": _metrics(reranked, queries, rows),
        "performance": {
            "load_seconds": round(load_seconds, 3),
            "pairs": len(pairs),
            "total_prediction_seconds": round(predict_seconds, 3),
            "ms_per_query": round(1_000 * predict_seconds / len(queries), 3),
            "ms_per_pair": round(1_000 * predict_seconds / len(pairs), 3),
        },
    }
    output = args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
