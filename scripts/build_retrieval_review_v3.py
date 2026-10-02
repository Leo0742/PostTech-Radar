from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

from app.core.config import DATABASE_PATH  # noqa: E402
from app.ml.v3.embeddings import DenseRetriever  # noqa: E402
from app.ml.v3.retrieval import LexicalRetriever, duplicate_free_ranking, minmax_normalize  # noqa: E402
from app.services.data_service import load_rows  # noqa: E402


def main() -> None:
    rows = [row for row in load_rows(DATABASE_PATH) if str(row.get("description") or "").strip()]
    texts = [str(row["description"]) for row in rows]
    groups = [str(row.get("normalized_description") or row["request_id"]) for row in rows]
    gte = json.loads((PROJECT_ROOT / "artifacts/research/v3/embeddings/gte-multilingual-base.json").read_text(encoding="utf-8"))
    root = PROJECT_ROOT / "artifacts/research/v3/embedding-cache"
    document = np.load(root / f"{gte['encoding']['document']['cache_key']}.npy", allow_pickle=False)
    query = np.load(root / f"{gte['encoding']['query']['cache_key']}.npy", allow_pickle=False)
    dense = DenseRetriever(document, query_embeddings=query)
    lexical = LexicalRetriever("word_char_tfidf").fit(texts)
    rng = np.random.default_rng(20260916)
    query_indices = np.linspace(0, len(rows) - 1, 20, dtype=int).tolist()
    query_indices.extend(int(value) for value in rng.choice(len(rows), size=10, replace=False))
    review = []
    for query_index in query_indices:
        lexical_scores = minmax_normalize(lexical.score(texts[query_index]))
        dense_scores = minmax_normalize(dense.score_index(query_index))
        methods = {
            "lexical": lexical_scores,
            "dense_gte": dense_scores,
            "hybrid": 0.5 * lexical_scores + 0.5 * dense_scores,
        }
        for method, scores in methods.items():
            candidate = duplicate_free_ranking(scores, query_index=query_index, groups=groups, limit=1)[0]
            review.append(
                {
                    "query_request_id": rows[query_index]["request_id"],
                    "query_text": texts[query_index],
                    "query_category_proxy": rows[query_index]["category"],
                    "method": method,
                    "candidate_request_id": rows[candidate]["request_id"],
                    "candidate_text": texts[candidate],
                    "candidate_category_proxy": rows[candidate]["category"],
                    "proxy_same_category": rows[candidate]["category"] == rows[query_index]["category"],
                    "score": round(float(scores[candidate]), 6),
                    "human_relevant_0_or_1": "",
                    "human_grade_0_to_3": "",
                    "reviewer_comment": "",
                }
            )
    output = PROJECT_ROOT / "outputs/retrieval_human_review_v3.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(review[0]))
        writer.writeheader()
        writer.writerows(review)
    print(output)


if __name__ == "__main__":
    main()
