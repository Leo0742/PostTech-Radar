# Model Card — PostTech Radar

## Что именно я обучал

В финальных 4B/8B вариантах я **не fine-tune'ил все веса Qwen**. Qwen3-Embedding использовался как frozen encoder, а я обучал собственную классификационную часть поверх embeddings и регистрационных признаков.

В проекте есть два deployment-профиля:

| Профиль | Encoder | Локальный artifact |
|---|---|---|
| **Lite / 4B** | Qwen3-Embedding-4B | `models/v5/category_qwen4b_lite.joblib` |
| **Quality / 8B** | Qwen3-Embedding-8B | `models/v5/category.joblib` |

В обоих pipeline использовались supervised classifier, prototype/kNN blend и отдельный TF-IDF specialist для сложных confusion pairs.

Моя часть работы: data audit, split protocol, baselines, embeddings, обучение классификаторов, tuning, leakage checks, сравнение кандидатов, PEFT experiments и интеграция модели в приложение.

## Данные

- **1 931** real labeled Service Desk tickets;
- **43** категории;
- synthetic rows в финальных deployment models: **0**.

Raw dataset в public repo не публикуется.

## Development results

| Pipeline | View | Top-1 | Top-3 | Macro-F1 |
|---|---|---:|---:|---:|
| **4B Lite** | TOP-15 | **80.781%** | **97.653%** | **76.008%** |
| **4B Lite** | FULL-43 | **73.276%** | **89.875%** | **55.292%** |
| **8B Quality V5.2** | TOP-15 | 79.883% | 97.136% | 74.007% |
| **8B Quality V5.2** | FULL-43 | 68.171% | 88.799% | 53.771% |

4B Lite и 8B Quality тюнились отдельно, поэтому эта таблица не является чистым benchmark размера encoder. Для matched сравнения см. [docs/training/MATCHED_QWEN4B_VS_QWEN8B.md](docs/training/MATCHED_QWEN4B_VS_QWEN8B.md).

У более раннего frozen 8B champion на отдельном V5 INTERNAL LOCKBOX TOP-15 было около **81.19% Top-1 / 98.02% Top-3 / 76.40% Macro-F1**. Этот lockbox не использовался для последующего tuning V5.2.

## 4B Lite

- model: `Qwen/Qwen3-Embedding-4B`;
- revision: `5cf2132abc99cad020ac570b19d031efec650f2b`;
- max length: 512;
- embedding dim: 2560;
- metadata scale: 0.75;
- kNN neighbors: 11;
- prototype temperature: 0.08;
- specialist: leakage-safe TF-IDF;
- real training rows: 1 931;
- artifact SHA-256: `ebd02b8d711b91a7b2ce63c30642d63a5e0a5a8320c2064f4be130c901550d26`.

Training/deployment builder: [scripts/v5_2_build_qwen4b_lite_deployment.py](scripts/v5_2_build_qwen4b_lite_deployment.py).

## 8B Quality

- model: `Qwen/Qwen3-Embedding-8B`;
- revision: `1d8ad4ca9b3dd8059ad90a75d4983776a23d44af`;
- max length: 512;
- embedding dim: 3072;
- supervised head: calibrated LinearSVC;
- blend: 0.90 supervised + 0.05 prototype + 0.05 kNN;
- specialist: leakage-safe TF-IDF;
- real training rows: 1 931;
- artifact SHA-256: `5c02b1337590c4e59bc2c0bfd5f279a3baee6a0bfe6945de568265afdf010e98`.

Training/deployment builder: [scripts/v5_2_build_deployment.py](scripts/v5_2_build_deployment.py).

## Что сравнивал

- TF-IDF + Logistic Regression;
- LinearSVC;
- CatBoost;
- Qwen3-Embedding-4B;
- Qwen3-Embedding-8B;
- prototype/kNN branches;
- TF-IDF specialist;
- synthetic-data screens;
- PEFT/LoRA experiments.

PEFT реально запускался, но стабильного выигрыша не дал и в deployment не вошёл. См. [docs/training/PEFT_SANITY.md](docs/training/PEFT_SANITY.md).

## Валидация и leakage

Я специально не использовал обычный random split как основной результат, потому что в датасете есть дубли и почти одинаковые обращения.

Frozen protocol:

- 1 830 unique duplicate groups;
- 3 repeats × 4 folds;
- group-disjoint train/validation;
- development: 1 241 rows;
- calibration: 306 rows;
- V5 INTERNAL LOCKBOX: 384 rows;
- group overlap audit: PASS;
- synthetic validation rows: 0.

Отдельно проверялись post-resolution поля и target-like признаки. Перед финальным измерением `routing_target` был удалён из specialist features.

Подробнее: [docs/training/V5_EVALUATION_PROTOCOL.md](docs/training/V5_EVALUATION_PROTOCOL.md).

## Почему `.joblib` bundles не лежат в public GitHub

Обученные artifacts реально были сохранены локально, и их hashes приведены выше. Но bundles строились на предоставленных Service Desk данных. Сериализованный TF-IDF vocabulary и другие структуры могут содержать производные от непубличных текстов.

Поэтому в public repo я оставляю то, что позволяет проверить мою работу без публикации чужих данных:

- training/deployment code;
- configs;
- split/leakage code;
- tests;
- aggregate reports;
- model hashes;
- inference/runtime code.

Основное описание обучения: [TRAINING.md](TRAINING.md).
