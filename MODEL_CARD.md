# Model Card — PostTech Radar

## Что именно я обучал

В проекте я не обучал Qwen3-Embedding-8B с нуля. Я использовал его как **frozen encoder** для получения embeddings и **обучал собственный классификационный pipeline** поверх этих признаков.

Финальный deployment recipe:

- encoder: **Qwen3-Embedding-8B** (frozen);
- trained head: **calibrated LinearSVC**;
- blend: **0.90 classifier + 0.05 prototype + 0.05 kNN**;
- дополнительный **TF-IDF specialist** для сложных пар категорий;
- structured metadata: service, component, request type, priority и другие регистрационные признаки;
- обучение на **1 931 реальном размеченном обращении**;
- synthetic training rows в финальной модели: **0**.

То есть моя работа здесь — это подготовка данных, построение признаков, обучение классификаторов, подбор архитектуры пайплайна, сравнение кандидатов, валидация и интеграция модели в приложение.

## Результаты

Для portfolio README я показываю leakage-safe cross-validation:

| View | Top-1 | Top-3 | Macro-F1 |
|---|---:|---:|---:|
| TOP-15 | **80.8%** | **97.7%** | **76.0%** |
| FULL-43 | **73.3%** | **89.9%** | **55.3%** |

В отдельном frozen lockbox evaluation для V5 champion был получен TOP-15 результат около **81.2% Top-1 / 98.0% Top-3 / 76.4% Macro-F1**. Lockbox не использовался для последующего подбора V5.2.

## Что сравнивал

В ходе экспериментов я проверял:

- TF-IDF + Logistic Regression;
- LinearSVC;
- CatBoost;
- Qwen3-Embedding-4B;
- Qwen3-Embedding-8B;
- prototype и kNN branches;
- specialist-модель на TF-IDF;
- PEFT/LoRA sanity experiments.

Финальный вариант выбирался по grouped cross-validation, а не по одному случайному split.

## Валидация

Я отдельно следил за утечками данных:

- duplicate/near-duplicate groups не пересекают train и validation;
- используется group-aware split;
- похожие обращения не должны одновременно попадать по разные стороны split;
- target-like поля исключались из specialist features;
- holdout/lockbox не использовался для подбора финальной конфигурации.

## Trained artifacts

Локально в проекте были сохранены обученные deployment artifacts, включая:

- `models/v5/category.joblib` — основной V5 deployment bundle;
- `models/v5/category_qwen4b_lite.joblib` — lite profile;
- классические `category.joblib`, `routing.joblib`, `retrieval.joblib`;
- evaluation metadata и model hashes.

Например, для одного из deployment classifiers в metadata сохранены:

- `trained_at: 2026-09-16T18:48:57Z`;
- `training_record_count: 1511` для соответствующего classical split;
- `model_artifact_sha256: c05e80b05186567c2e3ae408050b5ad5e85856345132024b861a300ee0130d56`.

Для финального V5.2 pipeline training использовались все **1 931** eligible real labeled tickets.

## Почему бинарные веса не лежат в public GitHub

Я специально не публикую `.joblib`/model bundles, потому что они обучены на предоставленных Service Desk данных. Сериализованные TF-IDF словари и другие структуры модели могут содержать производные от непубличного текста.

Поэтому публично оставлены:

- training/runtime code;
- model architecture and recipe;
- tests;
- dependency files;
- aggregate metrics;
- artifact metadata и hashes.

Так видно, что модель реально обучалась и использовалась, но при этом я не публикую чужие Service Desk данные или их производные.
