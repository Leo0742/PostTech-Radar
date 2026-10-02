from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, f1_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

from app.core.config import DATABASE_PATH  # noqa: E402
from app.ml.v3.dataset import challenge_dataset_config, training_corpus_summary  # noqa: E402
from app.ml.v3.protocol import manifest_from_dict  # noqa: E402
from app.services.data_service import load_rows  # noqa: E402

MODEL_ID = "cointegrated/rubert-tiny2"
REVISION = "e8ed3b0c8bbf4fb6984c3de043bf7d2f4e5969ae"


def measured_metrics(truth: list[str], predicted: list[str], probabilities: np.ndarray, labels: list[str]) -> dict:
    report = classification_report(truth, predicted, output_dict=True, zero_division=0)
    return {
        "accuracy": round(float(accuracy_score(truth, predicted)), 6),
        "macro_f1": round(float(f1_score(truth, predicted, average="macro", zero_division=0)), 6),
        "weighted_f1": round(float(f1_score(truth, predicted, average="weighted", zero_division=0)), 6),
        "worst_class_f1": round(
            min(float(report[label]["f1-score"]) for label in labels if label in report), 6
        ),
        "per_class": {
            label: {
                "precision": round(float(report[label]["precision"]), 6),
                "recall": round(float(report[label]["recall"]), 6),
                "f1": round(float(report[label]["f1-score"]), 6),
                "support": int(report[label]["support"]),
            }
            for label in labels
            if label in report
        },
        "probabilities_shape": list(probabilities.shape),
    }


def _data(database: Path):
    rows = load_rows(database)
    top = {item["name"] for item in training_corpus_summary(database, challenge_dataset_config())["top_k"]}
    rows = [row for row in rows if row["category"] in top and str(row["description"]).strip()]
    manifest = manifest_from_dict(
        json.loads((PROJECT_ROOT / "artifacts/evaluation/v3/protocol.json").read_text(encoding="utf-8"))
    )
    fold = manifest.folds[0]
    by_id = {str(row["request_id"]): row for row in rows}
    train = [by_id[value] for value in fold.train_request_ids]
    validation = [by_id[value] for value in fold.validation_request_ids]
    return manifest, fold, train, validation


def run_transformer(train, validation, labels, args):
    from datasets import Dataset
    from scipy.special import softmax
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DataCollatorWithPadding,
        Trainer,
        TrainingArguments,
    )

    label_to_id = {label: index for index, label in enumerate(labels)}
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, revision=args.revision, cache_dir=str(args.model_cache))
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_id,
        revision=args.revision,
        cache_dir=str(args.model_cache),
        num_labels=len(labels),
        id2label={index: label for label, index in label_to_id.items()},
        label2id=label_to_id,
    )
    def make_dataset(rows):
        dataset = Dataset.from_dict(
            {
                "text": [str(row["description"]) for row in rows],
                "label": [label_to_id[str(row["category"])] for row in rows],
            }
        )
        return dataset.map(
            lambda batch: tokenizer(batch["text"], truncation=True, max_length=128),
            batched=True,
            remove_columns=["text"],
        )
    train_dataset = make_dataset(train)
    validation_dataset = make_dataset(validation)
    training_args = TrainingArguments(
        output_dir=str(args.work_dir / "rubert-tiny2-finetune"),
        overwrite_output_dir=True,
        num_train_epochs=2,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        gradient_accumulation_steps=1,
        learning_rate=3e-5,
        weight_decay=0.01,
        warmup_ratio=0.1,
        eval_strategy="epoch",
        save_strategy="no",
        logging_strategy="epoch",
        report_to="none",
        use_cpu=True,
        seed=20260916,
        data_seed=20260916,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
    )
    started = time.perf_counter()
    trainer.train()
    train_seconds = time.perf_counter() - started
    started = time.perf_counter()
    output = trainer.predict(validation_dataset)
    predict_seconds = time.perf_counter() - started
    probabilities = softmax(output.predictions, axis=1)
    predicted = [labels[index] for index in probabilities.argmax(axis=1)]
    return predicted, probabilities, train_seconds, predict_seconds, {
        "strategy": "classifier_head_full_finetune",
        "epochs": 2,
        "learning_rate": 3e-5,
        "batch_size": 16,
        "max_length": 128,
    }


def run_setfit(train, validation, labels, args):
    from datasets import Dataset
    from setfit import SetFitModel, Trainer, TrainingArguments

    label_to_id = {label: index for index, label in enumerate(labels)}
    train_dataset = Dataset.from_dict(
        {
            "text": [str(row["description"]) for row in train],
            "label": [label_to_id[str(row["category"])] for row in train],
        }
    )
    model = SetFitModel.from_pretrained(
        args.model_id,
        revision=args.revision,
        cache_dir=str(args.model_cache),
        labels=labels,
    )
    model.model_body.max_seq_length = 128
    training_args = TrainingArguments(
        output_dir=str(args.work_dir / "setfit-rubert-tiny2"),
        batch_size=(16, 16),
        num_epochs=(1, 8),
        num_iterations=2,
        body_learning_rate=2e-5,
        head_learning_rate=0.01,
        sampling_strategy="oversampling",
        max_length=128,
        show_progress_bar=True,
        report_to="none",
        save_strategy="no",
        seed=20260916,
    )
    trainer = Trainer(model=model, args=training_args, train_dataset=train_dataset)
    started = time.perf_counter()
    trainer.train()
    train_seconds = time.perf_counter() - started
    texts = [str(row["description"]) for row in validation]
    started = time.perf_counter()
    probabilities = np.asarray(model.predict_proba(texts))
    predicted_raw = model.predict(texts)
    predict_seconds = time.perf_counter() - started
    predicted = [labels[int(value)] if isinstance(value, (int, np.integer)) else str(value) for value in predicted_raw]
    return predicted, probabilities, train_seconds, predict_seconds, {
        "strategy": "setfit_contrastive_plus_logistic_head",
        "body_epochs": 1,
        "head_epochs": 8,
        "num_iterations": 2,
        "batch_size": 16,
        "max_length": 128,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one resource-bounded supervised neural v3 experiment")
    parser.add_argument("mode", choices=("finetune", "setfit"))
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--revision", default=REVISION)
    parser.add_argument("--slug", default="rubert-tiny2")
    parser.add_argument("--database", type=Path, default=DATABASE_PATH)
    parser.add_argument("--model-cache", type=Path, default=PROJECT_ROOT / "work/model-cache-v3/huggingface")
    parser.add_argument("--work-dir", type=Path, default=PROJECT_ROOT / "work/neural-v3")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "artifacts/research/v3/neural")
    args = parser.parse_args()
    args.work_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(args.model_cache))
    os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", str(args.model_cache))
    manifest, fold, train, validation = _data(args.database)
    labels = list(manifest.labels)
    runner = run_transformer if args.mode == "finetune" else run_setfit
    predicted, probabilities, train_seconds, predict_seconds, configuration = runner(
        train, validation, labels, args
    )
    truth = [str(row["category"]) for row in validation]
    payload = {
        "candidate_id": f"category/{args.slug}-{args.mode}/v3",
        "status": "MEASURED_PARTIAL_ONE_GROUP_FOLD",
        "model_id": args.model_id,
        "revision": args.revision,
        "license": "MIT",
        "tier": "A",
        "dataset_sha256": manifest.dataset_sha256,
        "split_sha256": manifest.split_sha256,
        "sealed_holdout_accessed": False,
        "development_fold": {"repeat": fold.repeat, "fold": fold.fold, "train": len(train), "validation": len(validation)},
        "configuration": configuration,
        "train_seconds": round(train_seconds, 3),
        "prediction_ms_per_ticket": round(1_000 * predict_seconds / len(validation), 4),
        "metrics": measured_metrics(truth, predicted, probabilities, labels),
        "predictions": [
            {
                "request_id": str(row["request_id"]),
                "truth": truth[index],
                "prediction": predicted[index],
                "probabilities": {label: round(float(probabilities[index, position]), 8) for position, label in enumerate(labels)},
            }
            for index, row in enumerate(validation)
        ],
    }
    output = args.output_root / f"{args.slug}-{args.mode}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "metrics": payload["metrics"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
