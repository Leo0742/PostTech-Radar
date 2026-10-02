from __future__ import annotations

import argparse
import gc
import json
import math
import random
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from sklearn.preprocessing import OneHotEncoder


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.v5_dataset import DEFAULT_DATASET, load_v5_rows
from scripts.v5_experiment import (
    DEFAULT_CONTRACT,
    DEFAULT_PROTOCOL,
    labels_for_view,
    load_json,
    operator_metrics,
)
from scripts.v5_gpu_runner import (
    _fit_predict_supervised_head,
    _knn_probabilities,
    _normalize_rows,
    _prototype_probabilities,
    build_structured_features,
)


MODEL_ID = "Qwen/Qwen3-Embedding-8B"
MODEL_REVISION = "1d8ad4ca9b3dd8059ad90a75d4983776a23d44af"
INSTRUCTION_TEXT = (
    "Classify a Russian PostTech service-desk ticket by its operational request category. "
    "Focus on the concrete failure/action and distinguish close categories."
)


def _apply_instruction(text: str) -> str:
    return f"Instruct: {INSTRUCTION_TEXT}\nQuery: {text}"


def _load_oof_confusions() -> dict[str, list[str]]:
    root = ROOT / "artifacts/gpu_research_v5/runs/qwen8b_quality_sprint/quality_final_repeated"
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    for path in root.glob("*/result.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if payload.get("candidate", {}).get("view") != "top15":
            continue
        for item in payload.get("predictions", []):
            truth = str(item.get("truth") or "")
            top1 = str(item.get("top1") or "")
            if truth and top1 and truth != top1:
                counts[truth][top1] += 1
    return {label: [name for name, _ in counter.most_common(5)] for label, counter in counts.items()}


def _balanced_batch_indices(
    by_label: Mapping[str, Sequence[int]],
    rng: random.Random,
    *,
    classes_per_batch: int,
    samples_per_class: int,
) -> tuple[list[int], list[str]]:
    labels = [label for label, idxs in by_label.items() if len(idxs) >= 2]
    if len(labels) < 2:
        raise RuntimeError("Need at least two classes with >=2 real examples")
    chosen = rng.sample(labels, min(classes_per_batch, len(labels)))
    indexes: list[int] = []
    batch_labels: list[str] = []
    for label in chosen:
        pool = list(by_label[label])
        if len(pool) >= samples_per_class:
            take = rng.sample(pool, samples_per_class)
        else:
            take = [rng.choice(pool) for _ in range(samples_per_class)]
        indexes.extend(take)
        batch_labels.extend([label] * len(take))
    return indexes, batch_labels


def _supcon_loss(torch: Any, embeddings: Any, labels: Sequence[str], temperature: float) -> Any:
    z = torch.nn.functional.normalize(embeddings.float(), p=2, dim=1)
    sim = z @ z.T / temperature
    n = sim.shape[0]
    self_mask = torch.eye(n, dtype=torch.bool, device=sim.device)
    valid = ~self_mask
    sim_for_den = sim.masked_fill(~valid, float("-inf"))
    log_den = torch.logsumexp(sim_for_den, dim=1)
    losses = []
    for i, label in enumerate(labels):
        positive = torch.tensor(
            [j != i and labels[j] == label for j in range(n)],
            dtype=torch.bool,
            device=sim.device,
        )
        if positive.any():
            losses.append(-(sim[i, positive] - log_den[i]).mean())
    if not losses:
        raise RuntimeError("SupCon batch has no positive pairs")
    return torch.stack(losses).mean()


def _class_aware_mnrl_loss(torch: Any, embeddings: Any, labels: Sequence[str], temperature: float) -> Any:
    # Same-class examples are positives and are never placed in the negative set.
    z = torch.nn.functional.normalize(embeddings.float(), p=2, dim=1)
    sim = z @ z.T / temperature
    n = sim.shape[0]
    losses = []
    for i, label in enumerate(labels):
        positives = [j for j in range(n) if j != i and labels[j] == label]
        negatives = [j for j in range(n) if labels[j] != label]
        if not positives or not negatives:
            continue
        pos_score = torch.logsumexp(sim[i, positives], dim=0)
        den_score = torch.logsumexp(sim[i, positives + negatives], dim=0)
        losses.append(-(pos_score - den_score))
    if not losses:
        raise RuntimeError("Class-aware MNRL batch has no valid anchors")
    return torch.stack(losses).mean()


def _embed_training(model: Any, texts: Sequence[str], *, max_length: int) -> Any:
    features = model.tokenize([_apply_instruction(str(text)) for text in texts])
    features = {key: value.to(model.device) for key, value in features.items()}
    old = model.max_seq_length
    model.max_seq_length = max_length
    output = model(features)
    model.max_seq_length = old
    return output["sentence_embedding"]


def _encode_numpy(model: Any, texts: Sequence[str], *, batch_size: int, max_length: int) -> np.ndarray:
    old = model.max_seq_length
    model.max_seq_length = max_length
    values = model.encode(
        [_apply_instruction(str(text)) for text in texts],
        batch_size=batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    model.max_seq_length = old
    return np.asarray(values, dtype=np.float32)


def _load_model(rank: int, alpha: int, dropout: float, max_length: int) -> tuple[Any, dict[str, Any]]:
    import torch
    from peft import LoraConfig, TaskType, get_peft_model
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(
        MODEL_ID,
        revision=MODEL_REVISION,
        device="cuda",
        model_kwargs={"torch_dtype": torch.bfloat16},
    )
    model.max_seq_length = max_length
    transformer = model[0]
    base = transformer.auto_model
    base.gradient_checkpointing_enable()
    base.enable_input_require_grads()
    if hasattr(base.config, "use_cache"):
        base.config.use_cache = False
    peft = get_peft_model(
        base,
        LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            r=rank,
            lora_alpha=alpha,
            lora_dropout=dropout,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            bias="none",
        ),
    )
    transformer.auto_model = peft
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    total = sum(parameter.numel() for parameter in model.parameters())
    return model, {
        "trainable_parameters": trainable,
        "total_parameters": total,
        "trainable_fraction": trainable / total,
    }


def _sanity(
    rows: Sequence[dict[str, Any]],
    *,
    rank: int,
    alpha: int,
    dropout: float,
    lr: float,
    max_length: int,
    steps: int,
    seed: int,
) -> dict[str, Any]:
    import torch

    rng = random.Random(seed)
    by_label: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        by_label[str(row["category"])].append(index)
    eligible = [label for label, idxs in by_label.items() if len(idxs) >= 4]
    chosen = eligible[:4]
    sanity_rows: list[dict[str, Any]] = []
    for label in chosen:
        sanity_rows.extend(rows[index] for index in by_label[label][:4])
    if len(sanity_rows) < 8:
        raise RuntimeError("Not enough rows for PEFT sanity test")
    by_label_sanity: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(sanity_rows):
        by_label_sanity[str(row["category"])].append(index)

    model, stats = _load_model(rank, alpha, dropout, max_length)
    probe = [str(row["description"] or "") for row in sanity_rows[:8]]
    before = _encode_numpy(model, probe, batch_size=2, max_length=max_length)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=lr,
        weight_decay=0.0,
    )
    losses: list[float] = []
    first_grad_norm = 0.0
    model.train()
    for step in range(steps):
        indexes, labels = _balanced_batch_indices(
            by_label_sanity,
            rng,
            classes_per_batch=min(4, len(by_label_sanity)),
            samples_per_class=2,
        )
        texts = [str(sanity_rows[index]["description"] or "") for index in indexes]
        optimizer.zero_grad(set_to_none=True)
        embeddings = _embed_training(model, texts, max_length=max_length)
        loss = _supcon_loss(torch, embeddings, labels, 0.08)
        loss.backward()
        if step == 0:
            norms = [parameter.grad.detach().float().norm().item() for parameter in model.parameters() if parameter.grad is not None]
            first_grad_norm = float(sum(norms))
        torch.nn.utils.clip_grad_norm_([parameter for parameter in model.parameters() if parameter.requires_grad], 1.0)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    model.eval()
    after = _encode_numpy(model, probe, batch_size=2, max_length=max_length)
    cosine = np.sum(before * after, axis=1)
    drift = float(np.mean(1.0 - cosine))
    result = {
        **stats,
        "steps": steps,
        "loss_first": losses[0],
        "loss_last": losses[-1],
        "loss_min": min(losses),
        "first_grad_norm": first_grad_norm,
        "mean_embedding_cosine_drift": drift,
        "passed": bool(first_grad_norm > 0 and losses[-1] < losses[0] and drift > 1e-7),
    }
    del model, optimizer
    gc.collect()
    torch.cuda.empty_cache()
    return result


def _train_and_evaluate_fold(
    train_rows: Sequence[dict[str, Any]],
    validation_rows: Sequence[dict[str, Any]],
    labels: Sequence[str],
    confusion_targets: Mapping[str, Sequence[str]],
    *,
    rank: int,
    alpha: int,
    dropout: float,
    lr: float,
    train_max_length: int,
    eval_max_length: int,
    steps: int,
    objective: str,
    embedding_dim: int,
    seed: int,
) -> dict[str, Any]:
    import torch

    rng = random.Random(seed)
    by_label: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(train_rows):
        by_label[str(row["category"])].append(index)
    model, parameter_stats = _load_model(rank, alpha, dropout, train_max_length)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=lr,
        weight_decay=0.01,
    )
    losses: list[float] = []
    model.train()
    started = time.perf_counter()
    for step in range(steps):
        if objective in {"supcon", "class_aware_mnrl"}:
            indexes, batch_labels = _balanced_batch_indices(
                by_label,
                rng,
                classes_per_batch=3,
                samples_per_class=2,
            )
            texts = [str(train_rows[index]["description"] or "") for index in indexes]
            optimizer.zero_grad(set_to_none=True)
            embeddings = _embed_training(model, texts, max_length=train_max_length)
            if objective == "supcon":
                loss = _supcon_loss(torch, embeddings, batch_labels, 0.08)
            else:
                loss = _class_aware_mnrl_loss(torch, embeddings, batch_labels, 0.08)
        elif objective == "triplet":
            eligible = [label for label, idxs in by_label.items() if len(idxs) >= 2]
            anchors = rng.sample(eligible, min(2, len(eligible)))
            a_texts: list[str] = []
            p_texts: list[str] = []
            n_texts: list[str] = []
            for label in anchors:
                a_idx, p_idx = rng.sample(by_label[label], 2)
                negative_label = next(
                    (candidate for candidate in confusion_targets.get(label, []) if candidate in by_label and candidate != label),
                    None,
                )
                if negative_label is None:
                    negative_label = rng.choice([candidate for candidate in eligible if candidate != label])
                n_idx = rng.choice(by_label[negative_label])
                a_texts.append(str(train_rows[a_idx]["description"] or ""))
                p_texts.append(str(train_rows[p_idx]["description"] or ""))
                n_texts.append(str(train_rows[n_idx]["description"] or ""))
            optimizer.zero_grad(set_to_none=True)
            a = torch.nn.functional.normalize(_embed_training(model, a_texts, max_length=train_max_length).float(), dim=1)
            p = torch.nn.functional.normalize(_embed_training(model, p_texts, max_length=train_max_length).float(), dim=1)
            n = torch.nn.functional.normalize(_embed_training(model, n_texts, max_length=train_max_length).float(), dim=1)
            pos = (a * p).sum(dim=1)
            neg = (a * n).sum(dim=1)
            loss = torch.relu(0.15 + neg - pos).mean()
        else:
            raise ValueError(f"Unknown objective: {objective}")

        loss.backward()
        torch.nn.utils.clip_grad_norm_([parameter for parameter in model.parameters() if parameter.requires_grad], 1.0)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))

    train_seconds = time.perf_counter() - started
    model.eval()
    train_text = _encode_numpy(
        model,
        [str(row["description"] or "") for row in train_rows],
        batch_size=4,
        max_length=eval_max_length,
    )[:, :embedding_dim]
    validation_text = _encode_numpy(
        model,
        [str(row["description"] or "") for row in validation_rows],
        batch_size=4,
        max_length=eval_max_length,
    )[:, :embedding_dim]
    train_text = _normalize_rows(train_text)
    validation_text = _normalize_rows(validation_text)

    encoder = OneHotEncoder(handle_unknown="ignore", min_frequency=2, sparse_output=False)
    metadata_train = encoder.fit_transform(build_structured_features(train_rows))
    metadata_validation = encoder.transform(build_structured_features(validation_rows))
    x_train = np.concatenate([train_text, metadata_train], axis=1)
    x_validation = np.concatenate([validation_text, metadata_validation], axis=1)
    y_train = [str(row["category"]) for row in train_rows]
    truth = [str(row["category"]) for row in validation_rows]
    svc, effective = _fit_predict_supervised_head(
        "calibrated_linearsvc",
        x_train,
        y_train,
        x_validation,
        labels,
        seed=seed,
    )
    prototype = _prototype_probabilities(train_text, y_train, validation_text, labels)
    knn = _knn_probabilities(train_text, y_train, validation_text, labels)
    probabilities = 0.90 * svc + 0.05 * prototype + 0.05 * knn
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    metrics = operator_metrics(truth, probabilities, labels)
    result = {
        "metrics": metrics,
        "train_seconds": round(train_seconds, 3),
        "loss_first": losses[0],
        "loss_last": losses[-1],
        "loss_min": min(losses),
        "effective_head": effective + "+prototype_0.05+knn_0.05",
        **parameter_stats,
    }
    del model, optimizer
    gc.collect()
    torch.cuda.empty_cache()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="V5.2 Qwen3-Embedding-8B PEFT development-CV experiment")
    parser.add_argument("--mode", choices=("sanity", "fold"), required=True)
    parser.add_argument("--view", choices=("top15", "full43"), default="top15")
    parser.add_argument("--repeat", type=int, default=0)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--rank", type=int, default=16, choices=(8, 16, 32))
    parser.add_argument("--alpha", type=int)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--objective", choices=("supcon", "triplet", "class_aware_mnrl"), default="supcon")
    parser.add_argument("--train-max-length", type=int, default=256)
    parser.add_argument("--eval-max-length", type=int, default=512)
    parser.add_argument("--embedding-dim", type=int, default=3072, choices=(3072, 4096))
    parser.add_argument("--seed", type=int, default=20260918)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    alpha = args.alpha or args.rank * 2

    protocol = load_json(DEFAULT_PROTOCOL)
    contract = load_json(DEFAULT_CONTRACT)
    rows = load_v5_rows(DEFAULT_DATASET)
    lockbox_ids = set(str(value) for value in protocol["internal_lockbox_request_ids"])
    fold_spec = next(
        item
        for item in protocol["folds"]
        if int(item["repeat"]) == args.repeat and int(item["fold"]) == args.fold
    )
    if set(str(value) for value in fold_spec["train_request_ids"]) & lockbox_ids:
        raise RuntimeError("Training fold overlaps lockbox")
    if set(str(value) for value in fold_spec["validation_request_ids"]) & lockbox_ids:
        raise RuntimeError("Validation fold overlaps lockbox")

    labels = labels_for_view(contract, args.view)
    label_set = set(labels)
    rows_by_id = {str(row["request_id"]): row for row in rows}
    train_rows = [
        rows_by_id[str(value)]
        for value in fold_spec["train_request_ids"]
        if str(value) in rows_by_id and str(rows_by_id[str(value)]["category"]) in label_set
    ]
    validation_rows = [
        rows_by_id[str(value)]
        for value in fold_spec["validation_request_ids"]
        if str(value) in rows_by_id and str(rows_by_id[str(value)]["category"]) in label_set
    ]

    if args.mode == "sanity":
        payload: dict[str, Any] = {
            "status": "sanity",
            "internal_lockbox_accessed": False,
            "rank": args.rank,
            "alpha": alpha,
            "dropout": args.dropout,
            "learning_rate": args.learning_rate,
            "result": _sanity(
                train_rows,
                rank=args.rank,
                alpha=alpha,
                dropout=args.dropout,
                lr=args.learning_rate,
                max_length=args.train_max_length,
                steps=args.steps,
                seed=args.seed,
            ),
        }
    else:
        confusion_targets = _load_oof_confusions()
        result = _train_and_evaluate_fold(
            train_rows,
            validation_rows,
            labels,
            confusion_targets,
            rank=args.rank,
            alpha=alpha,
            dropout=args.dropout,
            lr=args.learning_rate,
            train_max_length=args.train_max_length,
            eval_max_length=args.eval_max_length,
            steps=args.steps,
            objective=args.objective,
            embedding_dim=args.embedding_dim,
            seed=args.seed + args.repeat * 100 + args.fold,
        )
        payload = {
            "status": "complete",
            "internal_lockbox_accessed": False,
            "view": args.view,
            "repeat": args.repeat,
            "fold": args.fold,
            "rank": args.rank,
            "alpha": alpha,
            "dropout": args.dropout,
            "learning_rate": args.learning_rate,
            "steps": args.steps,
            "objective": args.objective,
            "train_max_length": args.train_max_length,
            "eval_max_length": args.eval_max_length,
            "embedding_dim": args.embedding_dim,
            "result": result,
        }

    output = args.output
    if output is None:
        suffix = "sanity" if args.mode == "sanity" else f"r{args.repeat}f{args.fold}"
        output = ROOT / "artifacts/gpu_research_v5/runs/V5_2_QWEN8B_QUALITY_MAX/peft" / (
            f"{args.objective}_rank{args.rank}_{suffix}.json"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
