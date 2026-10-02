from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

from app.core.config import DATABASE_PATH  # noqa: E402
from app.ml.v3.retrieval import LexicalRetriever, evaluate_retriever  # noqa: E402
from app.services.data_service import load_rows  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark duplicate-free lexical retrieval on the full corpus")
    parser.add_argument("--database", type=Path, default=DATABASE_PATH)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "artifacts/research/v3/retrieval/lexical.json",
    )
    args = parser.parse_args()
    rows = [row for row in load_rows(args.database) if str(row.get("description") or "").strip()]
    results = []
    for kind in ("word_tfidf", "char_tfidf", "word_char_tfidf", "bm25"):
        retriever = LexicalRetriever(kind).fit([str(row["description"]) for row in rows])
        result = evaluate_retriever(retriever, rows)
        results.append(result)
        print(json.dumps({key: result[key] for key in ("model", "mrr", "hit_rate_at_5", "latency_ms_per_query")}, ensure_ascii=False))
    payload = {
        "status": "MEASURED",
        "protocol": "all non-empty descriptions; query self and normalized-description duplicates excluded",
        "relevance_proxy": "same category",
        "row_count": len(rows),
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
