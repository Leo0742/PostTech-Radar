from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for import_path in (ROOT, ROOT / "backend"):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from scripts.finalize_customer_adaptation import lite_fold_predict, load_all, topk_metrics
from scripts.gpu_customer_finetune import Config, _encode, _free_model, _train_encoder


CONFIG = Config("minilm_supcon_s120", 0, 2e-5, 120, "supcon", 256, 6)
FOLDS = ROOT / "artifacts/gpu_research_v5/protocol/folds.json"
PROTOCOL = ROOT / "artifacts/gpu_research_v5/protocol/protocol.json"
OUT = ROOT / "outputs/customer_gpu_finetune_2026-09-21/minilm_historical_supcon_s120.json"


def main() -> None:
    old_rows, new_rows, _old_minilm, _new_minilm, old_qwen, new_qwen = load_all()
    old_by_id = {str(row["request_id"]): row for row in old_rows}
    old_qwen_by_id = {str(row["request_id"]): old_qwen[i] for i, row in enumerate(old_rows)}
    new_ids = {str(row["request_id"]) for row in new_rows}
    labels = sorted({str(row["category"]) for row in old_rows + new_rows})
    lockbox = set(map(str, json.loads(PROTOCOL.read_text())["internal_lockbox_request_ids"]))
    folds = json.loads(FOLDS.read_text())
    records = []
    truths: list[str] = []
    probs: list[np.ndarray] = []

    for index, fold in enumerate(folds):
        train_ids = list(map(str, fold["train_request_ids"]))
        valid_ids = list(map(str, fold["validation_request_ids"]))
        if set(train_ids) & lockbox or set(valid_ids) & lockbox:
            raise RuntimeError("Frozen development fold touches lockbox")
        train_old = [old_by_id[value] for value in train_ids]
        valid_rows = [old_by_id[value] for value in valid_ids]
        train_rows = train_old + new_rows
        teacher_train = np.vstack([*[old_qwen_by_id[value] for value in train_ids], *new_qwen])
        seed = int(fold.get("seed", 20260921)) + 11000
        print(f"START minilm fold={index + 1}/12", flush=True)
        model, train_stats = _train_encoder("minilm", CONFIG, train_rows, new_ids, seed)
        model.eval()
        train_embeddings = _encode(model, "minilm", train_rows, 48)
        valid_embeddings = _encode(model, "minilm", valid_rows, 48)
        targets = [str(row["category"]) for row in train_rows]
        weights = np.concatenate([np.ones(len(train_old)), np.full(len(new_rows), 4.0)])
        probabilities = lite_fold_predict(
            train_rows,
            train_embeddings,
            teacher_train,
            targets,
            weights,
            valid_rows,
            valid_embeddings,
            labels,
        )
        truth = [str(row["category"]) for row in valid_rows]
        metrics = topk_metrics(truth, probabilities, labels)
        print(f"DONE minilm fold={index + 1}/12 {metrics}", flush=True)
        records.append({"index": index, "metrics": metrics, "train": train_stats})
        truths.extend(truth)
        probs.append(probabilities)
        _free_model(model)

    fold_mean = {
        key: float(np.mean([item["metrics"][key] for item in records]))
        for key in ("top1", "top3", "macro_f1_present_truth")
    }
    pooled = topk_metrics(truths, np.vstack(probs), labels)
    payload = {
        "method": "12 frozen V5 DEVELOPMENT folds; 66 customer rows train-only; fine-tuned MiniLM; no lockbox",
        "config": CONFIG.__dict__,
        "fold_mean": fold_mean,
        "pooled": pooled,
        "folds": records,
        "lockbox_accessed": False,
    }
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print("MINILM_HISTORICAL " + json.dumps(payload, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
