# Training notes

This file explains how I trained the category models used in PostTech Radar.

## What was trained

I used two Qwen embedding models as frozen encoders:

- `Qwen/Qwen3-Embedding-4B` — Lite profile;
- `Qwen/Qwen3-Embedding-8B` — Quality profile.

I did **not** fine-tune all Qwen weights for the final deployed models. The encoders produced embeddings, and I trained my own classification pipeline on top of them.

The final pipeline combines:

- a supervised classifier on embeddings and registration metadata;
- prototype scores;
- kNN scores;
- a small TF-IDF specialist for stable confusion pairs.

The blend is `0.90 supervised + 0.05 prototype + 0.05 kNN`.

## Data

The original project dataset contains **1,931 real labeled Service Desk tickets** and **43 categories**. The final deployment models use real labeled tickets only; synthetic training rows are `0`.

The raw Excel file is not published because it contains provided Service Desk data. The training code expects the dataset locally and keeps it outside Git.

## Validation

I used grouped validation because the dataset contains duplicate and near-duplicate tickets. Similar ticket groups are kept on one side of a split, so a repeated ticket cannot be in train and validation at the same time.

I also checked feature leakage. Fields created after ticket resolution are not used as inputs for category prediction.

For V5/V5.2 experiments the main selection was done on repeated grouped cross-validation. The frozen lockbox was kept separate from later tuning.

## 4B Lite

The final Lite profile uses:

- encoder: `Qwen3-Embedding-4B`;
- pinned revision: `5cf2132abc99cad020ac570b19d031efec650f2b`;
- max length: 512;
- embedding width: 2560;
- metadata scale: 0.75;
- kNN neighbors: 11;
- prototype temperature: 0.08;
- leakage-safe TF-IDF specialist.

Leakage-safe development results:

| View | Top-1 | Top-3 | Macro-F1 |
|---|---:|---:|---:|
| TOP-15 | **80.781%** | **97.653%** | **76.008%** |
| FULL-43 | **73.276%** | **89.875%** | **55.292%** |

Local deployment artifact from the original project:

`models/v5/category_qwen4b_lite.joblib`

SHA-256: `ebd02b8d711b91a7b2ce63c30642d63a5e0a5a8320c2064f4be130c901550d26`

## 8B Quality

The final Quality profile uses:

- encoder: `Qwen3-Embedding-8B`;
- pinned revision: `1d8ad4ca9b3dd8059ad90a75d4983776a23d44af`;
- max length: 512;
- embedding width: 3072;
- calibrated LinearSVC + prototype + kNN;
- leakage-safe TF-IDF specialist.

Leakage-safe development results for the final V5.2 recipe:

| View | Top-1 | Top-3 | Macro-F1 |
|---|---:|---:|---:|
| TOP-15 | **79.883%** | **97.136%** | **74.007%** |
| FULL-43 | **68.171%** | **88.799%** | **53.771%** |

The earlier frozen V5 lockbox result for TOP-15 was about **81.19% Top-1 / 98.02% Top-3 / 76.40% Macro-F1**. I keep this separate because the lockbox was not used to tune V5.2.

Local deployment artifact from the original project:

`models/v5/category.joblib`

SHA-256: `5c02b1337590c4e59bc2c0bfd5f279a3baee6a0bfe6945de568265afdf010e98`

## Experiments I compared

During the project I tested:

- TF-IDF + Logistic Regression;
- LinearSVC;
- CatBoost;
- Qwen3-Embedding-4B;
- Qwen3-Embedding-8B;
- prototype and kNN branches;
- TF-IDF specialist corrections;
- PEFT/LoRA sanity experiments.

The PEFT experiments were useful as a check, but they did not give a stable improvement over the frozen-encoder pipeline, so they were not selected for the final deployment.

## Why the `.joblib` bundles are not public

The public repository contains the training code, configs, model metadata, hashes and aggregate results. I do not publish the original dataset or serialized model bundles because they were built from provided Service Desk data and may contain vocabulary or other derived information from those texts.

The code under `scripts/` is the same training/evaluation path used in the original project. The reports under `docs/training/` preserve the final 4B/8B experiment results.
