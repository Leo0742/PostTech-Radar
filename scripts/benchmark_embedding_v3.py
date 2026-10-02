from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path

import psutil

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

from app.core.config import DATABASE_PATH  # noqa: E402
from app.ml.v3.dataset import challenge_dataset_config, training_corpus_summary  # noqa: E402
from app.ml.v3.embeddings import DenseRetriever, load_or_encode_embeddings  # noqa: E402
from app.ml.v3.experiments import evaluate_embedding_candidate  # noqa: E402
from app.ml.v3.protocol import manifest_from_dict  # noqa: E402
from app.ml.v3.retrieval import (  # noqa: E402
    LexicalRetriever,
    evaluate_dense_retriever,
    evaluate_hybrid_retriever,
)
from app.services.data_service import load_rows  # noqa: E402


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _category_rows(database: Path, rows: list[dict[str, object]]) -> list[dict[str, object]]:
    summary = training_corpus_summary(database, challenge_dataset_config())
    labels = {item["name"] for item in summary["top_k"]}
    return [row for row in rows if row["category"] in labels and str(row["description"]).strip()]


def _routing_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    allowed = {"(1 линия)", "(2 линия)", "(3 линия)"}
    return [
        {**row, "category": row["final_line"]}
        for row in rows
        if row["final_line"] in allowed and str(row["description"]).strip()
    ]


def _select_embeddings(matrix, all_rows, selected_rows):
    position = {str(row["request_id"]): index for index, row in enumerate(all_rows)}
    return matrix[[position[str(row["request_id"])] for row in selected_rows]]


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark one pinned sentence embedding model")
    parser.add_argument("model_id")
    parser.add_argument("revision")
    parser.add_argument("--slug", required=True)
    parser.add_argument("--query-prefix", default="")
    parser.add_argument("--document-prefix", default="")
    parser.add_argument("--classification-prefix", default="")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-seq-length", type=int, default=128)
    parser.add_argument("--database", type=Path, default=DATABASE_PATH)
    parser.add_argument("--model-cache", type=Path, default=PROJECT_ROOT / "work/model-cache-v3/huggingface")
    parser.add_argument("--embedding-cache", type=Path, default=PROJECT_ROOT / "artifacts/research/v3/embedding-cache")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "artifacts/research/v3/embeddings")
    args = parser.parse_args()

    os.environ.setdefault("HF_HOME", str(args.model_cache))
    os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", str(args.model_cache))
    from sentence_transformers import SentenceTransformer

    rows = [row for row in load_rows(args.database) if str(row.get("description") or "").strip()]
    texts = [str(row["description"]) for row in rows]
    process = psutil.Process()
    rss_before = process.memory_info().rss
    load_started = time.perf_counter()
    model = SentenceTransformer(
        args.model_id,
        revision=args.revision,
        cache_folder=str(args.model_cache),
        trust_remote_code=args.trust_remote_code,
        device="cpu",
    )
    model.max_seq_length = args.max_seq_length
    load_seconds = time.perf_counter() - load_started
    rss_loaded = process.memory_info().rss

    document_embeddings, document_metadata = load_or_encode_embeddings(
        model,
        texts,
        model_id=args.model_id,
        revision=args.revision,
        cache_dir=args.embedding_cache,
        prefix=args.document_prefix,
        batch_size=args.batch_size,
    )
    query_embeddings, query_metadata = load_or_encode_embeddings(
        model,
        texts,
        model_id=args.model_id,
        revision=args.revision,
        cache_dir=args.embedding_cache,
        prefix=args.query_prefix,
        batch_size=args.batch_size,
    )
    classification_embeddings, classification_metadata = load_or_encode_embeddings(
        model,
        texts,
        model_id=args.model_id,
        revision=args.revision,
        cache_dir=args.embedding_cache,
        prefix=args.classification_prefix,
        batch_size=args.batch_size,
    )
    rss_encoded = process.memory_info().rss

    dense = DenseRetriever(document_embeddings, query_embeddings=query_embeddings)
    dense_result = evaluate_dense_retriever(dense, rows, model=args.model_id)
    lexical = LexicalRetriever("word_char_tfidf").fit(texts)
    hybrid_results = [
        evaluate_hybrid_retriever(lexical, dense, rows, dense_weight=weight)
        for weight in (0.25, 0.5, 0.75)
    ]

    category_rows = _category_rows(args.database, rows)
    routing_rows = _routing_rows(rows)
    category_protocol = manifest_from_dict(
        json.loads((PROJECT_ROOT / "artifacts/evaluation/v3/protocol.json").read_text(encoding="utf-8"))
    )
    routing_protocol = manifest_from_dict(
        json.loads((PROJECT_ROOT / "artifacts/evaluation/v3/protocol_routing.json").read_text(encoding="utf-8"))
    )
    category_embeddings = _select_embeddings(classification_embeddings, rows, category_rows)
    routing_embeddings = _select_embeddings(classification_embeddings, rows, routing_rows)
    category_results = [
        evaluate_embedding_candidate(
            candidate_id=f"category/{args.slug}{'-metadata' if metadata else ''}/v3",
            model_id=args.model_id,
            revision=args.revision,
            embeddings=category_embeddings,
            rows=category_rows,
            manifest=category_protocol,
            with_metadata=metadata,
        )
        for metadata in (False, True)
    ]
    routing_results = [
        evaluate_embedding_candidate(
            candidate_id=f"routing/{args.slug}{'-metadata' if metadata else ''}/v3",
            model_id=args.model_id,
            revision=args.revision,
            embeddings=routing_embeddings,
            rows=routing_rows,
            manifest=routing_protocol,
            with_metadata=metadata,
        )
        for metadata in (False, True)
    ]
    payload = {
        "status": "MEASURED",
        "model_id": args.model_id,
        "revision": args.revision,
        "configuration": {
            "query_prefix": args.query_prefix,
            "document_prefix": args.document_prefix,
            "classification_prefix": args.classification_prefix,
            "trust_remote_code": args.trust_remote_code,
            "batch_size": args.batch_size,
            "max_seq_length": args.max_seq_length,
            "device": "cpu",
        },
        "environment": {
            "platform": platform.platform(),
            "load_seconds": round(load_seconds, 3),
            "rss_before_mb": round(rss_before / 1024**2, 1),
            "rss_after_load_mb": round(rss_loaded / 1024**2, 1),
            "rss_after_encode_mb": round(rss_encoded / 1024**2, 1),
        },
        "encoding": {
            "document": document_metadata,
            "query": query_metadata,
            "classification": classification_metadata,
        },
        "retrieval": dense_result,
        "hybrid_retrieval": hybrid_results,
        "category": category_results,
        "routing": routing_results,
    }
    output = args.output_root / f"{args.slug}.json"
    _write_json(output, payload)
    print(json.dumps({
        "output": str(output),
        "retrieval_mrr": dense_result["mrr"],
        "best_hybrid_mrr": max(item["mrr"] for item in hybrid_results),
        "category_macro_f1": [item["metrics"]["cv_macro_f1_mean"] for item in category_results],
        "routing_macro_f1": [item["metrics"]["cv_macro_f1_mean"] for item in routing_results],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
