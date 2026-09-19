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
  <img src="https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white" alt="FastAPI" />
  <img src="https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white" alt="scikit-learn" />
  <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch" />
  <img src="https://img.shields.io/badge/React-20232A?style=for-the-badge&logo=react&logoColor=61DAFB" alt="React" />
</p>

## About

> **Key point:** I prepared the data, trained the classifiers, and compared model variants myself. The final pipeline uses **Qwen3-Embedding-8B as a frozen encoder + a trained calibrated LinearSVC + prototype/kNN + a TF-IDF specialist**. It was trained on **1,931 real labeled tickets**. This is not just an LLM API integration.

[Model and training details](MODEL_CARD.md)

I built PostTech Radar as a student ML/NLP project for Service Desk tickets. The source dataset contained **1,931 labeled tickets**.

The goal is to help an operator understand where a new ticket should go. The service shows several likely categories, recommends a support line, and finds similar historical tickets. The operator still makes the final decision, and corrections can be stored as feedback for controlled retraining.

This is not an official PostTech product. I publish it as a portfolio project.

## What I worked on

- data import and validation;
- TOP-3 ticket category prediction;
- support-line recommendation;
- similar-ticket retrieval;
- comparison of **TF-IDF + Logistic Regression**, **LinearSVC**, **CatBoost**, and embedding-based approaches;
- experiments with **Qwen3 Embeddings** and sentence-transformers;
- group-disjoint validation and leakage checks;
- FastAPI backend and React/TypeScript frontend;
- operator feedback separated from automatic retraining.

## Two trained model variants

I built and compared **two embedding-based classifier variants**. In both cases Qwen was used as a frozen encoder, while the classification layer and the rest of the pipeline were trained on my labeled dataset.

| Profile | Encoder | What was trained | Purpose |
|---|---|---|---|
| **Lite** | Qwen3-Embedding-4B | classifier on top of embeddings + blend/specialist logic | lighter variant |
| **Quality** | Qwen3-Embedding-8B | classifier on top of embeddings + blend/specialist logic | main quality-focused variant |

Both variants used the same leakage-safe, group-aware validation approach. The final quality pipeline uses **Qwen3-Embedding-8B**.

> I did not fine-tune the 4B/8B Qwen weights themselves. I used them as encoders and **trained my own classifiers on top of the embeddings**. More details are in [MODEL_CARD.md](MODEL_CARD.md).

## Evaluation

Final metrics on leakage-safe grouped cross-validation:

| Classes | Top-1 | Top-3 | Macro-F1 |
|---|---:|---:|---:|
| TOP-15 | **80.8%** | **97.7%** | **76.0%** |
| FULL-43 | **73.3%** | **89.9%** | **55.3%** |

For me, the important part of this project was not only getting a metric, but also learning how to compare models on the same validation protocol and avoid train/validation leakage.

## Stack

`Python` · `NumPy` · `Pandas` · `scikit-learn` · `CatBoost` · `PyTorch` · `Transformers` · `sentence-transformers` · `FastAPI` · `React` · `TypeScript` · `pytest`

## Public repository note

The original Service Desk dataset, local database, trained model bundles, and generated ticket-level artifacts are intentionally not published. The repository contains the key application/ML source, tests, and dependency files needed to show the implementation without exposing the provided data.

See [PUBLIC_REPOSITORY_NOTICE.md](PUBLIC_REPOSITORY_NOTICE.md).
