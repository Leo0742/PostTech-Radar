<p align="right">
  <a href="README.md">Русский</a> · <b>English</b>
</p>

# PostTech Radar

<p align="center">
  <b>ML/NLP service for Service Desk ticket classification and routing</b><br />
  TOP-3 categories · support-line recommendation · similar-ticket search · operator feedback
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Qwen3--Embedding-4B-0F172A?style=for-the-badge" alt="Qwen3-Embedding-4B" />
  <img src="https://img.shields.io/badge/Qwen3--Embedding-8B-0F172A?style=for-the-badge" alt="Qwen3-Embedding-8B" />
  <img src="https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python" />
  <img src="https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white" alt="scikit-learn" />
  <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch" />
  <img src="https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white" alt="FastAPI" />\n  <a href="https://github.com/Leo0742/PostTech-Radar/actions/workflows/ci.yml"><img src="https://github.com/Leo0742/PostTech-Radar/actions/workflows/ci.yml/badge.svg" alt="CI" /></a>
</p>

## About

I built PostTech Radar as a student ML/NLP project for **1,931 labeled Service Desk tickets across 43 categories**.

The model gives an operator TOP-3 category suggestions for a new ticket. The service also recommends a support line and retrieves similar historical tickets.

> **What I actually trained:** I prepared the data, built leakage-safe splits, compared baselines, generated Qwen embeddings, trained classifiers on top of them, and evaluated separate 4B/8B pipelines. Qwen3-Embedding-4B/8B are frozen encoders in the final models; I do not present this as full Qwen fine-tuning.

Useful links:

- [training notes](TRAINING.md);
- [model card](MODEL_CARD.md);
- [4B Lite deployment builder](scripts/v5_2_build_qwen4b_lite_deployment.py);
- [8B Quality deployment builder](scripts/v5_2_build_deployment.py);
- [grouped evaluation protocol](scripts/v5_protocol.py);
- [data/leakage audit](scripts/v5_data_audit.py);
- [LoRA/PEFT sanity experiment](scripts/v5_2_peft_sanity.py);
- [training reports](docs/training/).

This is not an official PostTech product. I publish it as a student portfolio project.

## What I worked on

- data import and validation;
- TOP-3 text classification;
- TF-IDF + Logistic Regression, LinearSVC and CatBoost baselines;
- two embedding-based pipelines using **Qwen3-Embedding-4B** and **Qwen3-Embedding-8B**;
- structured metadata, prototype/kNN blending and a TF-IDF specialist for stable confusion pairs;
- group-disjoint validation and leakage checks;
- a PEFT/LoRA experiment for 8B, which I rejected because it did not show a stable gain;
- FastAPI backend and React/TypeScript frontend;
- operator feedback and controlled retraining.

## Two model profiles

Both models use Qwen as a **frozen embedding encoder**. The classifier on top of the embeddings is trained on my labeled data.

| Profile | Encoder | Main idea |
|---|---|---|
| **Lite / 4B** | Qwen3-Embedding-4B | lighter 2560-d pipeline |
| **Quality / 8B** | Qwen3-Embedding-8B | heavier 3072-d pipeline |

### Development metrics

| Pipeline | View | Top-1 | Top-3 | Macro-F1 |
|---|---|---:|---:|---:|
| **4B Lite** | TOP-15 | **80.781%** | **97.653%** | **76.008%** |
| **4B Lite** | FULL-43 | **73.276%** | **89.875%** | **55.292%** |
| **8B Quality (V5.2)** | TOP-15 | 79.883% | 97.136% | 74.007% |
| **8B Quality (V5.2)** | FULL-43 | 68.171% | 88.799% | 53.771% |

The two pipelines had separate tuning steps, so this is not a pure encoder-size benchmark. A matched 4B/8B reference is available [here](docs/training/MATCHED_QWEN4B_VS_QWEN8B.md).

An earlier frozen 8B champion reached **81.19% Top-1 / 98.02% Top-3 / 76.40% Macro-F1** on the separate V5 INTERNAL LOCKBOX for TOP-15. That lockbox was not used to tune V5.2.

## Evaluation protocol

The dataset contains repeated and near-duplicate tickets, so a random split could make the result look better than it really is.

The frozen grouped protocol contains:

- **1,931** real labeled rows;
- **1,830** unique duplicate groups;
- **3 repeats × 4 folds = 12** grouped folds;
- **1,241** development rows;
- **306** calibration rows;
- **384** V5 INTERNAL LOCKBOX rows;
- zero group overlap;
- zero synthetic rows in validation.

More details: [V5_EVALUATION_PROTOCOL.md](docs/training/V5_EVALUATION_PROTOCOL.md).

## PEFT / LoRA check

I also tested LoRA on Qwen3-Embedding-8B. The sanity run worked: about **15.34M trainable parameters (0.2022%)**, and the 20-step loss dropped from about **1.603 to 1.312**.

More serious PEFT checks did not produce a stable gain over the frozen-encoder pipeline, so PEFT was not selected for deployment.

Code: [scripts/v5_2_peft_sanity.py](scripts/v5_2_peft_sanity.py).

## Repository map

```text
scripts/
├── v5_data_audit.py
├── v5_dataset.py
├── v5_protocol.py
├── v5_experiment.py
├── v5_gpu_runner.py
├── v5_2_build_qwen4b_lite_deployment.py
├── v5_2_build_deployment.py
├── v5_2_peft_sanity.py
└── v5_2_benchmark_category_artifact.py

configs/v5/       public 4B/8B configs
docs/training/    evaluation and comparison reports
backend/app/ml/   inference/runtime
backend/tests/    data/split/runtime tests
```

## Stack

`Python` · `NumPy` · `Pandas` · `scikit-learn` · `CatBoost` · `PyTorch` · `Transformers` · `sentence-transformers` · `FastAPI` · `React` · `TypeScript` · `pytest`

## Checking the public repository

The public version can be checked without the private dataset and without a GPU:

```bash
git clone https://github.com/Leo0742/PostTech-Radar.git
cd PostTech-Radar

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

python -m compileall -q scripts backend/app backend/tests
python -m pytest backend/tests -q
```

These are the main checks used by GitHub Actions. Tests that require the original Service Desk XLSX are skipped in the public repository.

## Reproducing the 4B / 8B training

A local dataset with the same contract is required. The original Service Desk XLSX is not published.

Expected path:

```text
data/raw/Обращения_1931.xlsx
```

Build the data contract and grouped evaluation protocol first:

```bash
python scripts/v5_data_audit.py
python scripts/v5_protocol.py
```

GPU training requires NVIDIA CUDA. My setup used PyTorch 2.8 with CUDA 12.8:

```bash
python -m pip install torch==2.8.0 torchvision==0.23.0 \
  --index-url https://download.pytorch.org/whl/cu128

python -m pip install -r requirements-v5-gpu.txt
```

Then build the two deployment bundles:

```bash
# Qwen3-Embedding-4B Lite
python scripts/v5_2_build_qwen4b_lite_deployment.py

# Qwen3-Embedding-8B Quality
python scripts/v5_2_build_deployment.py
```

The scripts download the pinned Qwen encoders, compute embeddings, train the classifier pipeline, and save the local artifacts under `models/v5/`.

## Full web application

The full version I worked with used a local SQLite database, trained model bundles, and the original Service Desk data. Those private artifacts are not included here, so I do not claim that the public snapshot is a one-command full web demo.

The public repository is meant to make the ML/training code, validation protocol, runtime code, tests, and aggregate results reviewable without publishing private ticket data.

## Public repository note

I do not publish the original Service Desk XLSX, local database, ticket-level artifacts or serialized `.joblib` bundles.

The original project did contain real trained deployment artifacts:

- 4B Lite: `models/v5/category_qwen4b_lite.joblib`, SHA-256 `ebd02b8d711b91a7b2ce63c30642d63a5e0a5a8320c2064f4be130c901550d26`;
- 8B Quality: `models/v5/category.joblib`, SHA-256 `5c02b1337590c4e59bc2c0bfd5f279a3baee6a0bfe6945de568265afdf010e98`.

The binaries are omitted because they were built from provided Service Desk data and may contain vocabulary or other derived information. The public repo contains the training code, configs, hashes and aggregate evaluation reports instead.

See [PUBLIC_REPOSITORY_NOTICE.md](PUBLIC_REPOSITORY_NOTICE.md).
