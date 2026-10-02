from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

from app.core.config import DATABASE_PATH  # noqa: E402
from app.ml.v3.retrieval import LexicalRetriever, minmax_normalize  # noqa: E402
from app.services.data_service import load_rows  # noqa: E402


def _metrics(rankings: list[list[int]], query_indices: list[int], categories: list[str]) -> dict[str, float]:
    reciprocal: list[float] = []
    recalls = {5: [], 10: [], 20: []}
    ndcg: list[float] = []
    for query, ranking in zip(query_indices, rankings, strict=True):
        relevant = [categories[index] == categories[query] for index in ranking]
        positions = [position for position, value in enumerate(relevant, start=1) if value]
        reciprocal.append(1 / positions[0] if positions and positions[0] <= 10 else 0.0)
        for cutoff in recalls:
            recalls[cutoff].append(float(any(relevant[:cutoff])))
        gains = sum((1.0 / np.log2(position + 1)) for position in positions if position <= 10)
        ideal_count = min(10, sum(1 for value in categories if value == categories[query]) - 1)
        ideal = sum(1.0 / np.log2(position + 1) for position in range(1, ideal_count + 1))
        ndcg.append(float(gains / ideal) if ideal else 0.0)
    return {
        "mrr_at_10": round(float(np.mean(reciprocal)), 6),
        "recall_at_5": round(float(np.mean(recalls[5])), 6),
        "recall_at_10": round(float(np.mean(recalls[10])), 6),
        "recall_at_20": round(float(np.mean(recalls[20])), 6),
        "ndcg_at_10_proxy": round(float(np.mean(ndcg)), 6),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Sealed v4 dense/lexical retrieval tournament")
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--slug", required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--reranker", default="")
    parser.add_argument("--cache-dir", type=Path, default=Path(os.environ.get("HF_HOME", "/workspace/hf-cache")))
    args = parser.parse_args()

    import torch
    from huggingface_hub import model_info
    from sentence_transformers import SentenceTransformer

    pointer = json.loads((PROJECT_ROOT / "artifacts/gpu_research_v4/protocol.json").read_text(encoding="utf-8"))
    protocol = json.loads((PROJECT_ROOT / pointer["protocol_path"]).read_text(encoding="utf-8"))
    rows = [row for row in load_rows(DATABASE_PATH) if str(row.get("description") or "").strip()]
    texts = [str(row["description"]) for row in rows]
    categories = [str(row["category"]) for row in rows]
    groups = [str(row["normalized_description"]) for row in rows]
    by_id = {str(row["request_id"]): index for index, row in enumerate(rows)}
    dev_queries = [by_id[item] for item in protocol["retrieval_development_request_ids"] if item in by_id]
    test_queries = [by_id[item] for item in protocol["retrieval_test_request_ids"] if item in by_id]

    info = model_info(args.model_id)
    revision = str(info.sha)
    card = info.card_data.to_dict() if hasattr(info.card_data, "to_dict") else dict(info.card_data or {})
    model = SentenceTransformer(
        args.model_id,
        revision=revision,
        cache_folder=str(args.cache_dir),
        trust_remote_code=False,
        model_kwargs={"torch_dtype": torch.bfloat16},
    )
    instruction = None
    if "qwen3-embedding" in args.model_id.lower():
        instruction = "Instruct: Retrieve Russian Service Desk tickets about the same operational issue.\nQuery:"
    elif "e5-large-instruct" in args.model_id.lower():
        instruction = "Instruct: Given a Russian Service Desk ticket, retrieve tickets with the same issue.\nQuery:"
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    documents = np.asarray(
        model.encode(texts, batch_size=args.batch_size, normalize_embeddings=True, convert_to_numpy=True),
        dtype=np.float32,
    )
    queries = (
        np.asarray(
            model.encode(
                texts,
                batch_size=args.batch_size,
                normalize_embeddings=True,
                convert_to_numpy=True,
                prompt=instruction,
            ),
            dtype=np.float32,
        )
        if instruction
        else documents
    )
    encode_seconds = time.perf_counter() - started
    lexical = LexicalRetriever("word_char_tfidf").fit(texts)

    def scores(query: int) -> tuple[np.ndarray, np.ndarray]:
        lexical_scores = minmax_normalize(lexical.score(texts[query]))
        dense_scores = minmax_normalize(queries[query] @ documents.T)
        invalid = np.asarray([index == query or groups[index] == groups[query] for index in range(len(rows))])
        lexical_scores[invalid] = -1
        dense_scores[invalid] = -1
        return lexical_scores, dense_scores

    score_cache = {query: scores(query) for query in [*dev_queries, *test_queries]}

    def rankings(query_ids: list[int], weight: float) -> list[list[int]]:
        return [
            [int(index) for index in np.argsort(-((1 - weight) * score_cache[query][0] + weight * score_cache[query][1]))[:20]]
            for query in query_ids
        ]

    weights = [0.0, 0.25, 0.5, 0.75, 1.0]
    development_grid = {
        str(weight): _metrics(rankings(dev_queries, weight), dev_queries, categories) for weight in weights
    }
    selected_weight = max(weights, key=lambda weight: development_grid[str(weight)]["mrr_at_10"])
    test_rankings = rankings(test_queries, selected_weight)
    test_metrics = _metrics(test_rankings, test_queries, categories)

    reranker_result: dict[str, Any] | None = None
    if args.reranker:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        reranker_info = model_info(args.reranker)
        reranker_tokenizer = AutoTokenizer.from_pretrained(
            args.reranker,
            revision=str(reranker_info.sha),
            cache_dir=str(args.cache_dir),
            padding_side="left",
            trust_remote_code=False,
        )
        reranker_model = AutoModelForCausalLM.from_pretrained(
            args.reranker,
            revision=str(reranker_info.sha),
            cache_dir=str(args.cache_dir),
            torch_dtype=torch.bfloat16,
            device_map="cuda",
            trust_remote_code=False,
        ).eval()
        false_id = reranker_tokenizer.convert_tokens_to_ids("no")
        true_id = reranker_tokenizer.convert_tokens_to_ids("yes")

        def rerank(query_ids: list[int], base: list[list[int]], cutoff: int) -> list[list[int]]:
            result = []
            instruction = "Retrieve historical Russian Service Desk tickets that help resolve the same issue."
            prefix = (
                '<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the '
                'Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
            )
            suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
            for query, candidates in zip(query_ids, base, strict=True):
                active = candidates[:cutoff]
                prompts = [
                    prefix
                    + f"<Instruct>: {instruction}\n<Query>: {texts[query]}\n<Document>: {texts[index]}"
                    + suffix
                    for index in active
                ]
                scores = []
                for start in range(0, len(prompts), 8):
                    inputs = reranker_tokenizer(
                        prompts[start : start + 8],
                        padding=True,
                        truncation=True,
                        max_length=1024,
                        return_tensors="pt",
                    ).to(reranker_model.device)
                    with torch.inference_mode():
                        logits = reranker_model(**inputs).logits[:, -1, [false_id, true_id]]
                    scores.extend(torch.softmax(logits, dim=1)[:, 1].float().cpu().tolist())
                ordered = [active[int(index)] for index in np.argsort(-np.asarray(scores))]
                result.append([*ordered, *candidates[cutoff:]])
            return result

        # Select the reranker cutoff on a deterministic development subset;
        # the sealed test is touched only after the cutoff/accept decision.
        rerank_dev_queries = dev_queries[: min(200, len(dev_queries))]
        rerank_dev_base = rankings(rerank_dev_queries, selected_weight)
        dev_baseline = _metrics(rerank_dev_base, rerank_dev_queries, categories)
        dev_grid = {"0": dev_baseline}
        started = time.perf_counter()
        for cutoff in (10, 20):
            dev_grid[str(cutoff)] = _metrics(
                rerank(rerank_dev_queries, rerank_dev_base, cutoff), rerank_dev_queries, categories
            )
        selected_cutoff = max((0, 10, 20), key=lambda value: dev_grid[str(value)]["mrr_at_10"])
        reranked = rerank(test_queries, test_rankings, selected_cutoff) if selected_cutoff else test_rankings
        rerank_seconds = time.perf_counter() - started
        reranker_result = {
            "model_id": args.reranker,
            "revision": str(reranker_info.sha),
            "development_queries": len(rerank_dev_queries),
            "development_grid": dev_grid,
            "selected_cutoff": selected_cutoff,
            "metrics": _metrics(reranked, test_queries, categories),
            "seconds": round(rerank_seconds, 3),
            "latency_ms_per_query": round(1000 * rerank_seconds / len(test_queries), 4),
        }

    review = PROJECT_ROOT / "outputs/retrieval_human_review_v4.csv"
    with review.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=["query_id", "query", "rank", "candidate_id", "candidate", "human_label_0_3"],
        )
        writer.writeheader()
        for query, candidates in list(zip(test_queries, test_rankings, strict=True))[:20]:
            for rank, candidate in enumerate(candidates[:5], start=1):
                writer.writerow(
                    {
                        "query_id": rows[query]["request_id"],
                        "query": texts[query],
                        "rank": rank,
                        "candidate_id": rows[candidate]["request_id"],
                        "candidate": texts[candidate],
                        "human_label_0_3": "",
                    }
                )
    payload = {
        "status": "MEASURED_GPU_SEALED_RETRIEVAL",
        "model_id": args.model_id,
        "revision": revision,
        "license": str(card.get("license") or "UNKNOWN"),
        "dataset_sha256": protocol["dataset_sha256"],
        "split_sha256": protocol["split_sha256"],
        "development_queries": len(dev_queries),
        "sealed_test_queries": len(test_queries),
        "same_group_excluded": True,
        "selected_dense_weight": selected_weight,
        "development_grid": development_grid,
        "sealed_test": test_metrics,
        "reranker": reranker_result,
        "encode_seconds": round(encode_seconds, 3),
        "peak_vram_bytes": int(torch.cuda.max_memory_allocated()),
        "human_review_file": str(review.relative_to(PROJECT_ROOT)),
    }
    output = PROJECT_ROOT / "artifacts/gpu_research_v4/retrieval" / f"{args.slug}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "sealed_test": test_metrics, "reranker": reranker_result}, ensure_ascii=False))


if __name__ == "__main__":
    main()
