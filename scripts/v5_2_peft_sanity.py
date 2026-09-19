from __future__ import annotations

import argparse
import gc
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.v5_dataset import DEFAULT_DATASET, load_v5_rows

MODEL_ID = "Qwen/Qwen3-Embedding-8B"
MODEL_REVISION = "1d8ad4ca9b3dd8059ad90a75d4983776a23d44af"
INSTRUCTION_TEXT = (
    "Classify a Russian PostTech service-desk ticket by its operational request category. "
    "Focus on the concrete failure/action and distinguish close categories."
)


def _apply_instruction(text: str) -> str:
    return f"Instruct: {INSTRUCTION_TEXT}\nQuery: {text}"


def _balanced_batch_indices(
    by_label: dict[str, list[int]],
    rng: random.Random,
    *,
    classes_per_batch: int,
    samples_per_class: int,
) -> tuple[list[int], list[str]]:
    labels = [label for label, indexes in by_label.items() if len(indexes) >= 2]
    if len(labels) < 2:
        raise RuntimeError("Need at least two classes with >=2 examples")
    chosen = rng.sample(labels, min(classes_per_batch, len(labels)))
    indexes: list[int] = []
    batch_labels: list[str] = []
    for label in chosen:
        pool = by_label[label]
        taken = rng.sample(pool, samples_per_class) if len(pool) >= samples_per_class else [
            rng.choice(pool) for _ in range(samples_per_class)
        ]
        indexes.extend(taken)
        batch_labels.extend([label] * len(taken))
    return indexes, batch_labels


def _supcon_loss(torch: Any, embeddings: Any, labels: Sequence[str], temperature: float) -> Any:
    z = torch.nn.functional.normalize(embeddings.float(), p=2, dim=1)
    similarity = z @ z.T / temperature
    count = similarity.shape[0]
    self_mask = torch.eye(count, dtype=torch.bool, device=similarity.device)
    log_den = torch.logsumexp(similarity.masked_fill(self_mask, float("-inf")), dim=1)
    losses = []
    for index, label in enumerate(labels):
        positive = torch.tensor(
            [other != index and labels[other] == label for other in range(count)],
            dtype=torch.bool,
            device=similarity.device,
        )
        if positive.any():
            losses.append(-(similarity[index, positive] - log_den[index]).mean())
    if not losses:
        raise RuntimeError("SupCon batch has no positive pairs")
    return torch.stack(losses).mean()


def _embed_training(model: Any, texts: Sequence[str], *, max_length: int) -> Any:
    features = model.tokenize([_apply_instruction(str(text)) for text in texts])
    features = {key: value.to(model.device) for key, value in features.items()}
    previous = model.max_seq_length
    model.max_seq_length = max_length
    output = model(features)
    model.max_seq_length = previous
    return output["sentence_embedding"]


def _encode_numpy(model: Any, texts: Sequence[str], *, batch_size: int, max_length: int) -> np.ndarray:
    previous = model.max_seq_length
    model.max_seq_length = max_length
    values = model.encode(
        [_apply_instruction(str(text)) for text in texts],
        batch_size=batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    model.max_seq_length = previous
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
    peft_model = get_peft_model(
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
    transformer.auto_model = peft_model
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    total = sum(parameter.numel() for parameter in model.parameters())
    return model, {
        "trainable_parameters": trainable,
        "total_parameters": total,
        "trainable_fraction": trainable / total,
    }


def run_sanity(
    rows: list[dict[str, Any]],
    *,
    rank: int = 16,
    alpha: int = 32,
    dropout: float = 0.05,
    lr: float = 2e-5,
    max_length: int = 512,
    steps: int = 20,
    seed: int = 20260918,
) -> dict[str, Any]:
    import torch

    rng = random.Random(seed)
    by_label: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        by_label[str(row["category"])].append(index)

    eligible = [label for label, indexes in by_label.items() if len(indexes) >= 4]
    selected_labels = eligible[:4]
    sanity_rows: list[dict[str, Any]] = []
    for label in selected_labels:
        sanity_rows.extend(rows[index] for index in by_label[label][:4])
    if len(sanity_rows) < 8:
        raise RuntimeError("Not enough rows for PEFT sanity test")

    local_groups: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(sanity_rows):
        local_groups[str(row["category"])].append(index)

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
            local_groups,
            rng,
            classes_per_batch=min(4, len(local_groups)),
            samples_per_class=2,
        )
        texts = [str(sanity_rows[index]["description"] or "") for index in indexes]
        optimizer.zero_grad(set_to_none=True)
        embeddings = _embed_training(model, texts, max_length=max_length)
        loss = _supcon_loss(torch, embeddings, labels, 0.08)
        loss.backward()
        if step == 0:
            first_grad_norm = float(sum(
                parameter.grad.detach().float().norm().item()
                for parameter in model.parameters()
                if parameter.grad is not None
            ))
        torch.nn.utils.clip_grad_norm_(
            [parameter for parameter in model.parameters() if parameter.requires_grad], 1.0
        )
        optimizer.step()
        losses.append(float(loss.detach().cpu()))

    model.eval()
    after = _encode_numpy(model, probe, batch_size=2, max_length=max_length)
    drift = float(np.mean(1.0 - np.sum(before * after, axis=1)))

    result = {
        **stats,
        "rank": rank,
        "alpha": alpha,
        "dropout": dropout,
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Qwen8B LoRA/PEFT sanity experiment")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    rows = load_v5_rows(DEFAULT_DATASET)
    result = run_sanity(rows, steps=args.steps)
    text = json.dumps(result, ensure_ascii=False, indent=2)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
