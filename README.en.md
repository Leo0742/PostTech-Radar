# PostTech Radar

[Русский README](README.md)

Final application source for a Service Desk assistant: ticket import, category recommendations, support-line routing, historical retrieval, operator confirmation and SLA analytics.

The primary runtime uses a fine-tuned MiniLM encoder and classifier. Optional Qwen3-Embedding-4B rechecking uses a LoRA adapter. Original ticket data and trained binaries remain private, as explained in [PUBLIC_REPOSITORY_NOTICE.md](PUBLIC_REPOSITORY_NOTICE.md).

For full local use, supply your authorized `data/raw/` and `models/` files listed in [SUBMISSION_MANIFEST.md](SUBMISSION_MANIFEST.md), then follow [LOCAL_SETUP.md](docs/LOCAL_SETUP.md). GitHub does not provide these private files. The pinned base Qwen weights are downloaded from Hugging Face during setup.

Source-only checks need Python 3.12/3.13 and Node.js 22.12+:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pytest backend/tests -q
cd frontend
npm ci
npm test
npm run lint
npm run build
```

Tests requiring the original dataset are skipped when it is unavailable. Cold Qwen startup exceeded 180 seconds on the tested Mac, so successful end-to-end Qwen rechecking is not confirmed there. Primary MiniLM inference and the regular application flow were checked separately.

Existing TRAINING.md and MODEL_CARD.md describe earlier frozen Qwen V5.2 experiments, not the final fine-tuned runtime. This is a student portfolio project, not an official PostTech product.
