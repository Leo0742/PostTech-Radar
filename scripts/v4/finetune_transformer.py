from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from scipy.special import softmax
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.preprocessing import OneHotEncoder

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

from app.core.config import DATABASE_PATH  # noqa: E402
from app.ml.v4.evaluation import REGISTRATION_FIELDS, classification_metrics, safe_registration_row  # noqa: E402
from app.services.data_service import load_rows  # noqa: E402

METADATA_FIELDS = tuple(field for field in REGISTRATION_FIELDS if field != "description")


def _protocol() -> dict[str, Any]:
    pointer = json.loads((PROJECT_ROOT / "artifacts/gpu_research_v4/protocol.json").read_text(encoding="utf-8"))
    return json.loads((PROJECT_ROOT / pointer["protocol_path"]).read_text(encoding="utf-8"))


def _metadata_probabilities(train_rows: list[dict[str, Any]], validation_rows: list[dict[str, Any]], labels: list[str]) -> np.ndarray:
    encoder = OneHotEncoder(handle_unknown="ignore", min_frequency=2)
    train = np.asarray(
        [[safe_registration_row(row)[field] for field in METADATA_FIELDS] for row in train_rows], dtype=object
    )
    validation = np.asarray(
        [[safe_registration_row(row)[field] for field in METADATA_FIELDS] for row in validation_rows], dtype=object
    )
    train_matrix = encoder.fit_transform(train)
    validation_matrix = encoder.transform(validation)
    classifier = LogisticRegression(max_iter=2500, class_weight="balanced", random_state=20260917)
    classifier.fit(train_matrix, [str(row["category"]) for row in train_rows])
    raw = classifier.predict_proba(validation_matrix)
    aligned = np.zeros((len(validation_rows), len(labels)), dtype=np.float32)
    positions = {label: index for index, label in enumerate(labels)}
    for source, label in enumerate(classifier.classes_):
        aligned[:, positions[str(label)]] = raw[:, source]
    return aligned


def main() -> None:
    parser = argparse.ArgumentParser(description="GPU fine-tune a transformer on one frozen v4 fold")
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--revision", default="main")
    parser.add_argument("--slug", required=True)
    parser.add_argument("--mode", choices=("full", "lora"), required=True)
    parser.add_argument("--view", choices=("top15", "full43"), required=True)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260917)
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=192)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--early-stopping-patience", type=int, default=2)
    parser.add_argument("--loss", choices=("none", "balanced", "focal"), default="balanced")
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--metadata-weight", type=float, default=0.2)
    parser.add_argument("--run-tag", default="")
    parser.add_argument("--cache-dir", type=Path, default=Path(os.environ.get("HF_HOME", "/workspace/hf-cache")))
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "artifacts/gpu_research_v4/finetune")
    args = parser.parse_args()

    import torch
    from datasets import Dataset
    from huggingface_hub import model_info
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DataCollatorWithPadding,
        EarlyStoppingCallback,
        Trainer,
        TrainingArguments,
    )

    protocol = _protocol()
    all_rows = [row for row in load_rows(DATABASE_PATH) if str(row.get("description") or "").strip()]
    counts = Counter(str(row["category"]) for row in all_rows)
    selected_labels = {label for label, _count in counts.most_common(15)} if args.view == "top15" else set(counts)
    labels = sorted(selected_labels)
    label_to_id = {label: index for index, label in enumerate(labels)}
    rows_by_id = {str(row["request_id"]): row for row in all_rows if str(row["category"]) in selected_labels}
    fold = next(
        item for item in protocol["folds"] if int(item["repeat"]) == 0 and int(item["fold"]) == args.fold
    )
    train_rows = [rows_by_id[item] for item in fold["train_request_ids"] if item in rows_by_id]
    validation_rows = [rows_by_id[item] for item in fold["validation_request_ids"] if item in rows_by_id]

    info = model_info(args.model_id, revision=args.revision)
    revision = str(info.sha)
    card = info.card_data.to_dict() if hasattr(info.card_data, "to_dict") else dict(info.card_data or {})
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, revision=revision, cache_dir=str(args.cache_dir))
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_id,
        revision=revision,
        cache_dir=str(args.cache_dir),
        num_labels=len(labels),
        id2label={index: label for label, index in label_to_id.items()},
        label2id=label_to_id,
        torch_dtype=torch.bfloat16,
    )
    if args.mode == "lora":
        from peft import LoraConfig, TaskType, get_peft_model

        model = get_peft_model(
            model,
            LoraConfig(
                task_type=TaskType.SEQ_CLS,
                r=16,
                lora_alpha=32,
                lora_dropout=0.05,
                target_modules=["query", "key", "value"],
                modules_to_save=["classifier"],
            ),
        )
    model.gradient_checkpointing_enable()
    input_require_grads_enabled = False
    if args.mode == "lora":
        # Gradient checkpointing needs at least one grad-requiring input to
        # propagate through the frozen base model into PEFT adapters.
        model.enable_input_require_grads()
        input_require_grads_enabled = True
    model.config.use_cache = False

    def make_dataset(rows: list[dict[str, Any]]) -> Dataset:
        dataset = Dataset.from_dict(
            {
                "text": [str(row["description"]) for row in rows],
                "label": [label_to_id[str(row["category"])] for row in rows],
            }
        )
        return dataset.map(
            lambda batch: tokenizer(batch["text"], truncation=True, max_length=args.max_length),
            batched=True,
            remove_columns=["text"],
        )

    train_dataset = make_dataset(train_rows)
    validation_dataset = make_dataset(validation_rows)
    suffix = f"-{args.run_tag}" if args.run_tag else ""
    run_dir = PROJECT_ROOT / "work/v4-finetune" / f"{args.slug}-{args.mode}-{args.view}-f{args.fold}-s{args.seed}{suffix}"
    learning_rate = args.learning_rate or (2e-5 if args.mode == "full" else 8e-5)
    training_args = TrainingArguments(
        output_dir=str(run_dir),
        overwrite_output_dir=True,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=max(2, args.batch_size * 2),
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        logging_strategy="steps",
        logging_steps=20,
        report_to="none",
        bf16=True,
        gradient_checkpointing=True,
        seed=args.seed,
        data_seed=args.seed,
        dataloader_num_workers=2,
    )
    train_counts = Counter(str(row["category"]) for row in train_rows)
    class_weights = torch.tensor(
        [len(train_rows) / (len(labels) * max(1, train_counts[label])) for label in labels],
        dtype=torch.float32,
    )

    class WeightedTrainer(Trainer):
        def compute_loss(self, model: Any, inputs: dict[str, Any], return_outputs: bool = False, **kwargs: Any) -> Any:
            labels_tensor = inputs.pop("labels")
            outputs = model(**inputs)
            weights = class_weights.to(outputs.logits.device) if args.loss != "none" else None
            losses = torch.nn.functional.cross_entropy(outputs.logits, labels_tensor, weight=weights, reduction="none")
            if args.loss == "focal":
                probabilities = torch.softmax(outputs.logits, dim=-1)
                target_probability = probabilities.gather(1, labels_tensor.unsqueeze(1)).squeeze(1)
                losses = ((1 - target_probability) ** args.focal_gamma) * losses
            loss = losses.mean()
            return (loss, outputs) if return_outputs else loss

    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer, pad_to_multiple_of=8),
        compute_metrics=lambda result: {
            "macro_f1": float(
                f1_score(result.label_ids, np.asarray(result.predictions).argmax(axis=1), average="macro", zero_division=0)
            )
        },
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience)],
    )
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    trainer.train()
    train_seconds = time.perf_counter() - started
    started = time.perf_counter()
    output = trainer.predict(validation_dataset)
    prediction_seconds = time.perf_counter() - started
    text_probabilities = softmax(output.predictions, axis=1)
    metadata_probabilities = _metadata_probabilities(train_rows, validation_rows, labels)
    combined_probabilities = (1 - args.metadata_weight) * text_probabilities + args.metadata_weight * metadata_probabilities
    truth = [str(row["category"]) for row in validation_rows]
    predictions = {
        "text": [labels[index] for index in text_probabilities.argmax(axis=1)],
        "metadata": [labels[index] for index in metadata_probabilities.argmax(axis=1)],
        "text_metadata_80_20": [labels[index] for index in combined_probabilities.argmax(axis=1)],
    }
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    total = sum(parameter.numel() for parameter in model.parameters())
    payload = {
        "candidate_id": f"category/{args.slug}-{args.mode}/{args.view}/v4",
        "status": "MEASURED_GPU_FROZEN_FOLD",
        "model_id": args.model_id,
        "revision": revision,
        "license": str(card.get("license") or "UNKNOWN"),
        "trust_remote_code": False,
        "mode": args.mode,
        "view": args.view,
        "fold": args.fold,
        "seed": args.seed,
        "run_tag": args.run_tag,
        "hyperparameters": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "gradient_accumulation": args.gradient_accumulation,
            "learning_rate": learning_rate,
            "weight_decay": args.weight_decay,
            "warmup_ratio": args.warmup_ratio,
            "max_length": args.max_length,
            "loss": args.loss,
            "focal_gamma": args.focal_gamma,
            "metadata_weight": args.metadata_weight,
            "early_stopping_patience": args.early_stopping_patience,
        },
        "dataset_sha256": protocol["dataset_sha256"],
        "split_sha256": protocol["split_sha256"],
        "sealed_holdout_accessed": False,
        "real_only_evaluation": True,
        "training_integrity": {
            "gradient_checkpointing": True,
            "input_require_grads_enabled": input_require_grads_enabled,
            "valid_for_neural_summary": args.mode != "lora" or input_require_grads_enabled,
        },
        "train_rows": len(train_rows),
        "validation_rows": len(validation_rows),
        "parameters_total": total,
        "parameters_trainable": trainable,
        "train_seconds": round(train_seconds, 3),
        "prediction_ms_per_ticket": round(1000 * prediction_seconds / len(validation_rows), 4),
        "peak_vram_bytes": int(torch.cuda.max_memory_allocated()),
        "metrics": {
            name: classification_metrics(truth, values, labels=labels) for name, values in predictions.items()
        },
        "predictions": [
            {
                "request_id": str(row["request_id"]),
                "truth": truth[index],
                **{name: values[index] for name, values in predictions.items()},
            }
            for index, row in enumerate(validation_rows)
        ],
    }
    result = args.output_root / f"{args.slug}-{args.mode}-{args.view}-f{args.fold}-s{args.seed}{suffix}.json"
    result.parent.mkdir(parents=True, exist_ok=True)
    result.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(result), "metrics": payload["metrics"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
