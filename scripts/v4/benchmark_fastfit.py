from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

from app.core.config import DATABASE_PATH  # noqa: E402
from app.ml.v4.evaluation import classification_metrics  # noqa: E402
from app.services.data_service import load_rows  # noqa: E402


def _fastfit_examples(rows: list[dict[str, object]]) -> dict[str, list[str]]:
    return {
        "text": [str(row["description"]) for row in rows],
        "label": [str(row["category"]) for row in rows],
    }


def _fastfit_splits(
    train_rows: list[dict[str, object]], validation_rows: list[dict[str, object]]
) -> dict[str, dict[str, list[str]]]:
    validation = _fastfit_examples(validation_rows)
    return {
        "train": _fastfit_examples(train_rows),
        "validation": validation,
        # FastFit 1.2.1 requires a test split for every custom dataset even
        # when prediction is disabled. This alias is never used for scoring.
        "test": validation,
    }


def _train_fastfit_compat(trainer: Any) -> Any:
    # FastFit 1.2.1 passes a set here, while the pinned Transformers
    # evaluation loop concatenates this value with a list.
    return trainer.trainer.train(
        resume_from_checkpoint=trainer.checkpoint,
        ignore_keys_for_eval=["doc_input_ids", "doc_attention_mask", "labels"],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark IBM FastFit on one frozen v4 fold")
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--slug", required=True)
    parser.add_argument("--view", choices=("top15", "full43"), required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260917)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=192)
    parser.add_argument("--run-tag", default="")
    args = parser.parse_args()

    import torch
    from datasets import Dataset, DatasetDict
    from fastfit import FastFitTrainer
    from huggingface_hub import model_info

    pointer = json.loads((PROJECT_ROOT / "artifacts/gpu_research_v4/protocol.json").read_text(encoding="utf-8"))
    protocol = json.loads((PROJECT_ROOT / pointer["protocol_path"]).read_text(encoding="utf-8"))
    fold = next(
        item for item in protocol["folds"] if int(item["repeat"]) == 0 and int(item["fold"]) == args.fold
    )
    all_rows = [row for row in load_rows(DATABASE_PATH) if str(row.get("description") or "").strip()]
    counts = Counter(str(row["category"]) for row in all_rows)
    labels = sorted({label for label, _ in counts.most_common(15)} if args.view == "top15" else set(counts))
    label_to_id = {label: index for index, label in enumerate(labels)}
    by_id = {str(row["request_id"]): row for row in all_rows if str(row["category"]) in label_to_id}
    train_rows = [by_id[item] for item in fold["train_request_ids"] if item in by_id]
    validation_rows = [by_id[item] for item in fold["validation_request_ids"] if item in by_id]
    dataset = DatasetDict(
        {name: Dataset.from_dict(examples) for name, examples in _fastfit_splits(train_rows, validation_rows).items()}
    )
    info = model_info(args.model_id)
    torch.cuda.reset_peak_memory_stats()
    trainer = FastFitTrainer(
        model_name_or_path=args.model_id,
        dataset=dataset,
        label_column_name="label",
        text_column_name="text",
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=max(16, args.batch_size * 2),
        max_text_length=args.max_length,
        dataloader_drop_last=False,
        num_repeats=args.repeats,
        optim="adafactor",
        learning_rate=args.learning_rate,
        clf_loss_factor=0.1,
        fp16=True,
        do_train=True,
        do_eval=True,
        evaluation_strategy="epoch",
        save_strategy="no",
        report_to="none",
        output_dir=str(
            PROJECT_ROOT
            / "work/v4-fastfit"
            / f"{args.slug}-{args.view}-f{args.fold}-s{args.seed}-{args.run_tag or 'base'}"
        ),
        overwrite_output_dir=True,
        seed=args.seed,
    )
    started = time.perf_counter()
    _train_fastfit_compat(trainer)
    train_seconds = time.perf_counter() - started
    prediction = trainer.trainer.predict(trainer.eval_dataset)
    predicted_ids = np.asarray(prediction.predictions).argmax(axis=1)
    predicted = [labels[int(index)] for index in predicted_ids]
    truth = [str(row["category"]) for row in validation_rows]
    payload = {
        "candidate_id": f"category/fastfit-{args.slug}/{args.view}/v4",
        "status": "MEASURED_GPU_FROZEN_FOLD",
        "package": "fast-fit==1.2.1",
        "model_id": args.model_id,
        "revision": str(info.sha),
        "view": args.view,
        "fold": args.fold,
        "seed": args.seed,
        "hyperparameters": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "num_repeats": args.repeats,
            "max_length": args.max_length,
        },
        "train_rows": len(train_rows),
        "validation_rows": len(validation_rows),
        "train_seconds": round(train_seconds, 3),
        "peak_vram_bytes": int(torch.cuda.max_memory_allocated()),
        "metrics": classification_metrics(truth, predicted, labels=labels),
        "predictions": [
            {
                "request_id": str(row["request_id"]),
                "truth": truth[index],
                "prediction": predicted[index],
            }
            for index, row in enumerate(validation_rows)
        ],
    }
    suffix = f"-{args.run_tag}" if args.run_tag else ""
    output = (
        PROJECT_ROOT
        / "artifacts/gpu_research_v4/fastfit"
        / f"{args.slug}-{args.view}-f{args.fold}-s{args.seed}{suffix}.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "metrics": payload["metrics"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
