#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"
export HF_HOME="$ROOT_DIR/work/model-cache-v3/huggingface"
export SENTENCE_TRANSFORMERS_HOME="$HF_HOME"

stage="${1:-all}"

if [[ "$stage" == "classical" || "$stage" == "all" ]]; then
  .venv-v3/bin/python scripts/evaluate_v3.py
  .venv-v3/bin/python scripts/train_candidate_v3.py all --task category
  .venv-v3/bin/python scripts/train_candidate_v3.py all --task routing
  .venv-v3/bin/python scripts/benchmark_retrieval_v3.py
fi

if [[ "$stage" == "embeddings" || "$stage" == "all" ]]; then
  .venv-v3/bin/python scripts/benchmark_embedding_v3.py Alibaba-NLP/gte-multilingual-base 9bbca17d9273fd0d03d5725c7a4b0f6b45142062 --slug gte-multilingual-base --trust-remote-code --batch-size 32
  .venv-v3/bin/python scripts/benchmark_embedding_v3.py intfloat/multilingual-e5-base d128750597153bb5987e10b1c3493a34e5a4502a --slug multilingual-e5-base --query-prefix 'query: ' --document-prefix 'passage: ' --classification-prefix 'query: ' --batch-size 32
  .venv-v3/bin/python scripts/benchmark_embedding_v3.py ai-forever/ru-en-RoSBERTa 89fb1651989adbb1cfcfdedafd7d102951ad0555 --slug ru-en-rosberta --query-prefix 'search_query: ' --document-prefix 'search_document: ' --classification-prefix 'classification: ' --batch-size 24
fi

if [[ "$stage" == "neural" || "$stage" == "all" ]]; then
  .venv-v3/bin/python scripts/train_neural_candidate_v3.py finetune
  .venv-v3/bin/python scripts/train_neural_candidate_v3.py setfit
  .venv-v3/bin/python scripts/benchmark_reranker_v3.py
fi

if [[ "$stage" == "final" || "$stage" == "all" ]]; then
  .venv-v3/bin/python scripts/finalize_v3.py
  .venv-v3/bin/python scripts/finalize_routing_v3.py
  .venv-v3/bin/python scripts/smoke_v3.py
  .venv-v3/bin/python scripts/build_retrieval_review_v3.py
fi
