from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.finalize_customer_adaptation import load_all, qwen_fold_predict, topk_metrics
from scripts.gpu_customer_finetune import OUT_DIR, _new_folds


def main() -> None:
    old_rows, new_rows, _old_minilm, _new_minilm, old_qwen, new_qwen = load_all()
    labels = sorted({str(row["category"]) for row in old_rows + new_rows})
    base_by_id: dict[str, np.ndarray] = {}

    for train_new_idx, valid_new_idx in _new_folds(new_rows):
        train_new = [new_rows[index] for index in train_new_idx]
        valid_rows = [new_rows[index] for index in valid_new_idx]
        train_rows = old_rows + train_new
        train_embeddings = np.vstack([old_qwen, new_qwen[train_new_idx]])
        train_targets = [str(row["category"]) for row in train_rows]
        weights = np.concatenate([
            np.ones(len(old_rows), dtype=float),
            np.full(len(train_new), 16.0, dtype=float),
        ])
        probabilities = qwen_fold_predict(
            train_rows,
            train_embeddings,
            train_targets,
            weights,
            valid_rows,
            new_qwen[valid_new_idx],
            labels,
        )
        for row, probs in zip(valid_rows, probabilities, strict=True):
            base_by_id[str(row["request_id"])] = np.asarray(probs, dtype=float)

    fine = json.loads((OUT_DIR / "qwen_r16_mnrl_s80.json").read_text(encoding="utf-8"))
    fine_labels = [str(value) for value in fine["labels"]]
    if fine_labels != labels:
        raise RuntimeError("Label ordering mismatch")
    fine_by_id = {
        str(item["request_id"]): np.asarray(item["probabilities"], dtype=float)
        for item in fine["predictions"]
    }
    ids = [str(row["request_id"]) for row in new_rows]
    truth = [str(row["category"]) for row in new_rows]
    base = np.vstack([base_by_id[value] for value in ids])
    tuned = np.vstack([fine_by_id[value] for value in ids])

    rows = []
    for tuned_weight in np.linspace(0.0, 1.0, 21):
        probs = (1.0 - tuned_weight) * base + tuned_weight * tuned
        probs /= probs.sum(axis=1, keepdims=True)
        rows.append({
            "tuned_weight": round(float(tuned_weight), 2),
            **topk_metrics(truth, probs, labels),
        })

    baseline = rows[0]
    pareto = [
        row for row in rows
        if row["top1"] >= baseline["top1"]
        and row["top3"] >= baseline["top3"]
        and row["macro_f1_present_truth"] >= baseline["macro_f1_present_truth"]
    ]
    payload = {
        "method": "same 2-fold customer OOF; blend only, no refit; no lockbox",
        "baseline": baseline,
        "fine_tuned": rows[-1],
        "pareto_vs_baseline": pareto,
        "grid": rows,
        "lockbox_accessed": False,
    }
    output = OUT_DIR / "qwen_base_finetune_blend.json"
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
