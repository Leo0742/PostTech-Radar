from __future__ import annotations

import argparse
import gc
import json
import random
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for import_path in (ROOT, ROOT / "backend"):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from scripts.finalize_customer_adaptation import (  # noqa: E402
    load_all,
    lite_fold_predict,
    qwen_fold_predict,
    topk_metrics,
)
from scripts.v5_experiment import EXTENDED_INSTRUCTION_REGISTRY  # noqa: E402


SEED = 20260921
QWEN_ID = "Qwen/Qwen3-Embedding-4B"
QWEN_REVISION = "5cf2132abc99cad020ac570b19d031efec650f2b"
QWEN_INSTRUCTION = EXTENDED_INSTRUCTION_REGISTRY["posttech_tight_c"]
MINILM_ID = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
OUT_DIR = ROOT / "outputs/customer_gpu_finetune_2026-09-21"


@dataclass(frozen=True)
class Config:
    name: str
    rank: int
    lr: float
    steps: int
    objective: str
    max_length: int
    customer_repeat: int


QWEN_CONFIGS = [
    Config("qwen_r8_supcon_s50", 8, 2e-5, 50, "supcon", 256, 6),
    Config("qwen_r16_mnrl_s80", 16, 2e-5, 80, "mnrl", 256, 8),
    Config("qwen_r16_mnrl_s50_lr1e5", 16, 1e-5, 50, "mnrl", 256, 8),
    Config("qwen_r16_mnrl_s60_lr15e6", 16, 1.5e-5, 60, "mnrl", 256, 8),
]
MINILM_CONFIGS = [
    Config("minilm_supcon_s120", 0, 2e-5, 120, "supcon", 256, 6),
    Config("minilm_mnrl_s180", 0, 4e-5, 180, "mnrl", 256, 8),
]


def _normalize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return np.divide(values, norms, out=np.zeros_like(values), where=norms > 0)


def _new_folds(rows: Sequence[dict[str, Any]]) -> list[tuple[list[int], list[int]]]:
    by_label: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        by_label[str(row["category"])].append(index)
    validation = [[], []]
    for label in sorted(by_label):
        for offset, index in enumerate(by_label[label]):
            validation[offset % 2].append(index)
    universe = set(range(len(rows)))
    return [(sorted(universe - set(valid)), sorted(valid)) for valid in validation]


def _pooled_metrics(truths: list[str], probabilities: list[np.ndarray], labels: Sequence[str]) -> dict[str, float]:
    if not probabilities:
        raise RuntimeError("No validation probabilities were collected")
    return topk_metrics(truths, np.vstack(probabilities), labels)


def _batch_indices(
    rows: Sequence[dict[str, Any]],
    customer_ids: set[str],
    rng: random.Random,
    *,
    customer_repeat: int,
    classes_per_batch: int = 4,
    samples_per_class: int = 2,
) -> tuple[list[int], list[str]]:
    pools: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        repeats = customer_repeat if str(row["request_id"]) in customer_ids else 1
        pools[str(row["category"])].extend([index] * max(1, repeats))
    eligible = [label for label, pool in pools.items() if len(set(pool)) >= 2]
    if len(eligible) < 2:
        raise RuntimeError("Need at least two labels with two distinct examples")
    chosen = rng.sample(eligible, min(classes_per_batch, len(eligible)))
    indexes: list[int] = []
    labels: list[str] = []
    for label in chosen:
        unique = list(dict.fromkeys(pools[label]))
        if len(unique) >= samples_per_class:
            # Weighted choice without replacement when customer rows are present.
            weighted = pools[label]
            first = rng.choice(weighted)
            second_pool = [value for value in weighted if value != first]
            second = rng.choice(second_pool)
            take = [first, second]
        else:
            take = unique
        indexes.extend(take)
        labels.extend([label] * len(take))
    return indexes, labels


def _contrastive_loss(torch: Any, embeddings: Any, labels: Sequence[str], objective: str) -> Any:
    z = torch.nn.functional.normalize(embeddings.float(), p=2, dim=1)
    sim = z @ z.T / 0.08
    n = int(sim.shape[0])
    losses = []
    for i, label in enumerate(labels):
        positives = [j for j in range(n) if j != i and labels[j] == label]
        negatives = [j for j in range(n) if labels[j] != label]
        if not positives or not negatives:
            continue
        if objective == "supcon":
            candidates = positives + negatives
            den = torch.logsumexp(sim[i, candidates], dim=0)
            losses.append(-(sim[i, positives] - den).mean())
        elif objective == "mnrl":
            pos = torch.logsumexp(sim[i, positives], dim=0)
            den = torch.logsumexp(sim[i, positives + negatives], dim=0)
            losses.append(-(pos - den))
        else:
            raise ValueError(objective)
    if not losses:
        raise RuntimeError("Contrastive batch has no valid anchors")
    return torch.stack(losses).mean()


def _load_encoder(model_kind: str, config: Config) -> Any:
    import torch
    from sentence_transformers import SentenceTransformer

    if model_kind == "qwen":
        from peft import LoraConfig, TaskType, get_peft_model

        model = SentenceTransformer(
            QWEN_ID,
            revision=QWEN_REVISION,
            trust_remote_code=True,
            device="cuda",
            model_kwargs={"torch_dtype": torch.bfloat16},
        )
        model.max_seq_length = config.max_length
        transformer = model[0]
        base = transformer.auto_model
        base.gradient_checkpointing_enable()
        base.enable_input_require_grads()
        if hasattr(base.config, "use_cache"):
            base.config.use_cache = False
        transformer.auto_model = get_peft_model(
            base,
            LoraConfig(
                task_type=TaskType.FEATURE_EXTRACTION,
                r=config.rank,
                lora_alpha=config.rank * 2,
                lora_dropout=0.05,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                bias="none",
            ),
        )
        return model
    if model_kind == "minilm":
        model = SentenceTransformer(MINILM_ID, device="cuda")
        model.max_seq_length = config.max_length
        return model
    raise ValueError(model_kind)


def _prepared_texts(model_kind: str, rows: Sequence[dict[str, Any]]) -> list[str]:
    values = [" ".join(str(row.get("description") or "").split()) for row in rows]
    if model_kind == "qwen":
        return [f"Instruct: {QWEN_INSTRUCTION}\nQuery: {value}" for value in values]
    return values


def _embed_train(model: Any, texts: Sequence[str]) -> Any:
    features = model.tokenize(list(texts))
    features = {key: value.to(model.device) for key, value in features.items()}
    return model(features)["sentence_embedding"]


def _encode(model: Any, model_kind: str, rows: Sequence[dict[str, Any]], batch_size: int) -> np.ndarray:
    values = model.encode(
        _prepared_texts(model_kind, rows),
        batch_size=batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    values = np.asarray(values, dtype=np.float32)
    if model_kind == "qwen":
        values = values[:, :2560]
    return _normalize(values)


def _train_encoder(
    model_kind: str,
    config: Config,
    train_rows: Sequence[dict[str, Any]],
    customer_ids: set[str],
    seed: int,
) -> tuple[Any, dict[str, float]]:
    import torch

    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    rng = random.Random(seed)
    model = _load_encoder(model_kind, config)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=config.lr, weight_decay=0.01)
    losses: list[float] = []
    model.train()
    started = time.perf_counter()
    for step in range(config.steps):
        indexes, labels = _batch_indices(
            train_rows,
            customer_ids,
            rng,
            customer_repeat=config.customer_repeat,
        )
        batch_rows = [train_rows[index] for index in indexes]
        optimizer.zero_grad(set_to_none=True)
        embeddings = _embed_train(model, _prepared_texts(model_kind, batch_rows))
        loss = _contrastive_loss(torch, embeddings, labels, config.objective)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
        if (step + 1) % 10 == 0 or step == 0:
            print(
                f"TRAIN {model_kind} {config.name} step={step + 1}/{config.steps} loss={losses[-1]:.5f}",
                flush=True,
            )
    return model, {
        "seconds": round(time.perf_counter() - started, 3),
        "loss_first": losses[0],
        "loss_last": losses[-1],
        "loss_min": min(losses),
    }


def _free_model(model: Any) -> None:
    import torch

    del model
    gc.collect()
    torch.cuda.empty_cache()


def evaluate_config(model_kind: str, config: Config) -> dict[str, Any]:
    old_rows, new_rows, old_minilm, new_minilm, old_qwen, new_qwen = load_all()
    labels = sorted({str(row["category"]) for row in old_rows + new_rows})
    folds = _new_folds(new_rows)
    all_truth: list[str] = []
    all_probabilities: list[np.ndarray] = []
    all_predictions: list[dict[str, Any]] = []
    fold_results = []

    for fold_number, (train_new_idx, valid_new_idx) in enumerate(folds):
        fold_seed = SEED + fold_number * 100 + (1 if model_kind == "qwen" else 2)
        train_new = [new_rows[index] for index in train_new_idx]
        valid_rows = [new_rows[index] for index in valid_new_idx]
        train_rows = old_rows + train_new
        customer_train_ids = {str(row["request_id"]) for row in train_new}
        model, train_stats = _train_encoder(model_kind, config, train_rows, customer_train_ids, fold_seed)
        model.eval()
        batch_size = 6 if model_kind == "qwen" else 48
        train_embeddings = _encode(model, model_kind, train_rows, batch_size)
        valid_embeddings = _encode(model, model_kind, valid_rows, batch_size)
        if model_kind == "qwen":
            OUT_DIR.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                OUT_DIR / f"{config.name}_fold{fold_number}_embeddings.npz",
                train_embeddings=train_embeddings,
                valid_embeddings=valid_embeddings,
                train_ids=np.asarray([str(row["request_id"]) for row in train_rows]),
                valid_ids=np.asarray([str(row["request_id"]) for row in valid_rows]),
            )
        targets = [str(row["category"]) for row in train_rows]
        truth = [str(row["category"]) for row in valid_rows]
        weights = np.concatenate(
            [
                np.ones(len(old_rows), dtype=float),
                np.full(len(train_new), 16.0 if model_kind == "qwen" else 4.0, dtype=float),
            ]
        )
        if model_kind == "qwen":
            probabilities = qwen_fold_predict(
                train_rows,
                train_embeddings,
                targets,
                weights,
                valid_rows,
                valid_embeddings,
                labels,
            )
        else:
            teacher_train = np.vstack([old_qwen, new_qwen[train_new_idx]])
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
        metrics = topk_metrics(truth, probabilities, labels)
        fold_results.append({"fold": fold_number, "metrics": metrics, "train": train_stats})
        all_truth.extend(truth)
        all_probabilities.append(probabilities)
        for row, target, row_probabilities in zip(valid_rows, truth, probabilities, strict=True):
            all_predictions.append(
                {
                    "request_id": str(row["request_id"]),
                    "truth": target,
                    "probabilities": [float(value) for value in row_probabilities],
                }
            )
        print(f"EVAL {model_kind} {config.name} fold={fold_number} {metrics}", flush=True)
        _free_model(model)

    return {
        "model": model_kind,
        "config": asdict(config),
        "method": "2-fold category-stratified customer OOF; historical 1931 train-only; no lockbox",
        "metrics": _pooled_metrics(all_truth, all_probabilities, labels),
        "labels": labels,
        "predictions": all_predictions,
        "folds": fold_results,
        "lockbox_accessed": False,
    }


def run_sweep(model_kind: str, config_name: str | None = None) -> dict[str, Any]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    configs = QWEN_CONFIGS if model_kind == "qwen" else MINILM_CONFIGS
    if config_name:
        configs = [config for config in configs if config.name == config_name]
        if not configs:
            raise ValueError(f"Unknown config: {config_name}")
    results = []
    for config in configs:
        result = evaluate_config(model_kind, config)
        results.append(result)
        (OUT_DIR / f"{config.name}.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    winner = max(
        results,
        key=lambda item: (
            float(item["metrics"]["top1"]),
            float(item["metrics"]["top3"]),
            float(item["metrics"]["macro_f1_present_truth"]),
        ),
    )
    payload = {"model": model_kind, "results": results, "winner": winner, "lockbox_accessed": False}
    (OUT_DIR / f"{model_kind}_sweep.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("SWEEP_WINNER " + json.dumps(winner, ensure_ascii=False), flush=True)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=("qwen", "minilm"), required=True)
    parser.add_argument("--config")
    args = parser.parse_args()
    run_sweep(args.model, args.config)


if __name__ == "__main__":
    main()
