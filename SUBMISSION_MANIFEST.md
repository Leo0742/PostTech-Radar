> Этот список описывает полный ZIP в [приватном релизе](https://github.com/Leo0742/PostTech-Radar/releases/tag/complete-backup-20261002-120850). Данные и бинарные веса находятся в архиве релиза; Git LFS для его скачивания не требуется.

# Состав сдачи

В репозитории находятся исходный код приложения, оба Excel-файла из задания, скрипты подготовки данных, тесты и финальные дообученные модели. Виртуальные окружения, `node_modules`, SQLite-база, frontend-сборка, исследовательские кэши и базовые веса Hugging Face в Git не попадают.

## Модели, которые использует приложение

| Файл | Назначение | SHA-256 |
|---|---|---|
| `models/customer_2026-09-21/category_lite_finetuned.joblib` | Финальная голова основного MiniLM-классификатора | `b8d2c02c5eef0f71bf6d5b62836718f6c064c25313a5428db3032788600c99bc` |
| `models/customer_2026-09-21/minilm_supcon_s120/model.safetensors` | Дообученные веса MiniLM | `a4a489fc60bfc820678a44d606e084fa2ead4a751c588af8252a384977c94ab3` |
| `models/customer_2026-09-21/category_qwen_finetuned.joblib` | Финальная голова Qwen для ручной перепроверки | `447236e6627c808f5960c0368253e92a045e9bf77a3190d62504cd766afa3192` |
| `models/customer_2026-09-21/qwen_r16_mnrl_s80_adapter/adapter_model.safetensors` | Наш LoRA-адаптер Qwen3-Embedding-4B | `c6422a1b9eeb00583e419f211defcf25aa0074f63032afcfe38685f1e427b063` |
| `models/category.joblib` | Локальный fallback категории | `e72aa031463125cdb03050a34685967a5f7031709c8b385418cef8b64564cff5` |
| `models/routing.joblib` | Рекомендация линии поддержки | `35cf99f8e05eabc5e113ba7bcdaa9ae9c39cef291a097b655f0950808d0ab532` |
| `models/retrieval.joblib` | Поиск похожих обращений | `52cd19904edc269c31706345d0e26af7be671acc6f80add21c1260b2eca1267c` |

Файлы `*.safetensors` хранятся через Git LFS. Базовый `Qwen/Qwen3-Embedding-4B` не дублируется в GitLab: `scripts/setup.py` скачивает ревизию `5cf2132abc99cad020ac570b19d031efec650f2b` с Hugging Face, после чего приложение автоматически накладывает наш LoRA-адаптер.

## Данные

| Файл | SHA-256 |
|---|---|
| `data/raw/Обращения_1931.xlsx` | `daf742da6bb5902149585a05a3e38a2c655347cf6001f9cae640e1f27069ac40` |
| `data/raw/ALL_TICKETS.xlsx` | `9611a2fb9d3334c1e4c946d36528218c7f75456542e524fea2f313d64c69a3d6` |

`data/processed/posttech.db` создаётся при установке и специально не хранится в репозитории.
