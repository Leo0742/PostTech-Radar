from __future__ import annotations

import ast
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from rapidfuzz.fuzz import ratio

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

from app.core.config import DATABASE_PATH  # noqa: E402
from app.ml.v3.candidates import CandidateSpec, build_estimator  # noqa: E402
from app.ml.v4.evaluation import classification_metrics  # noqa: E402
from app.services.data_service import load_rows  # noqa: E402


def _parse_list(value: str) -> list[str]:
    start, end = value.find("["), value.rfind("]")
    if start < 0 or end <= start:
        return []
    try:
        parsed = json.loads(value[start : end + 1])
    except json.JSONDecodeError:
        try:
            parsed = ast.literal_eval(value[start : end + 1])
        except (ValueError, SyntaxError):
            return []
    return [str(item).strip() for item in parsed if isinstance(item, str) and 20 <= len(item.strip()) <= 1500]


def main() -> None:
    import torch
    from huggingface_hub import model_info
    from sentence_transformers import SentenceTransformer
    from transformers import AutoModelForCausalLM, AutoTokenizer

    pointer = json.loads((PROJECT_ROOT / "artifacts/gpu_research_v4/protocol.json").read_text(encoding="utf-8"))
    protocol = json.loads((PROJECT_ROOT / pointer["protocol_path"]).read_text(encoding="utf-8"))
    fold = next(item for item in protocol["folds"] if int(item["repeat"]) == 0 and int(item["fold"]) == 0)
    all_rows = [row for row in load_rows(DATABASE_PATH) if str(row.get("description") or "").strip()]
    by_id = {str(row["request_id"]): row for row in all_rows}
    train_rows = [by_id[item] for item in fold["train_request_ids"]]
    validation_rows = [by_id[item] for item in fold["validation_request_ids"]]
    definitions = yaml.safe_load((PROJECT_ROOT / "data/category_definitions_v4.yaml").read_text(encoding="utf-8"))
    definition_by_label = {str(item["category"]): item for item in definitions}
    support = Counter(str(row["category"]) for row in train_rows)
    targets = [label for label in protocol["labels"] if support[label] <= 20]

    generator_id = "Qwen/Qwen3-8B"
    generator_info = model_info(generator_id)
    tokenizer = AutoTokenizer.from_pretrained(generator_id, revision=generator_info.sha, cache_dir=os.environ.get("HF_HOME"))
    generator = AutoModelForCausalLM.from_pretrained(
        generator_id,
        revision=generator_info.sha,
        cache_dir=os.environ.get("HF_HOME"),
        torch_dtype=torch.bfloat16,
        device_map="cuda",
    )
    generated: list[dict[str, Any]] = []
    for position, label in enumerate(targets):
        definition = definition_by_label[label]
        requested = min(50, max(10, support[label] * (10 if support[label] <= 5 else 5)))
        prompt = (
            "Ты генерируешь реалистичные обезличенные обращения российского Service Desk. "
            f"Категория: {label}. Определение: {definition['definition']}. "
            f"Термины: {', '.join(definition['typical_symptoms'][:10])}. "
            f"Примеры только для стиля: {json.dumps(definition['canonical_train_examples'][:3], ensure_ascii=False)}. "
            f"Верни только JSON-массив из {requested} разных текстов. Добавляй опечатки, сокращения, короткие и длинные формулировки; "
            "не копируй примеры, не добавляй персональные данные и не упоминай название категории."
        )
        messages = [{"role": "user", "content": prompt}]
        encoded = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt", enable_thinking=False
        ).to("cuda")
        torch.manual_seed(20260917 + position)
        with torch.inference_mode():
            output = generator.generate(
                encoded,
                max_new_tokens=1800,
                do_sample=True,
                temperature=0.8,
                top_p=0.9,
                repetition_penalty=1.05,
            )
        decoded = tokenizer.decode(output[0, encoded.shape[1] :], skip_special_tokens=True)
        for text in _parse_list(decoded):
            generated.append(
                {
                    "description": text,
                    "category": label,
                    "generator_model": generator_id,
                    "generator_revision": str(generator_info.sha),
                    "prompt_version": "v4.1",
                    "seed": 20260917 + position,
                    "source_category": label,
                }
            )
    del generator
    torch.cuda.empty_cache()

    real_texts = [" ".join(str(row["description"]).lower().split()) for row in all_rows]
    rule_filtered = []
    seen: list[str] = []
    for sample in generated:
        normalized = " ".join(sample["description"].lower().split())
        if any(ratio(normalized, candidate) >= 90 for candidate in real_texts + seen):
            continue
        if len(set(normalized.split())) < 4:
            continue
        seen.append(normalized)
        rule_filtered.append(sample)

    verifier = SentenceTransformer(
        "Qwen/Qwen3-Embedding-0.6B",
        cache_folder=os.environ.get("HF_HOME"),
        model_kwargs={"torch_dtype": torch.bfloat16},
    )
    label_texts = [definition_by_label[label]["definition"] for label in protocol["labels"]]
    label_embeddings = verifier.encode(label_texts, normalize_embeddings=True, convert_to_numpy=True)
    sample_embeddings = verifier.encode(
        [sample["description"] for sample in rule_filtered],
        batch_size=8,
        normalize_embeddings=True,
        convert_to_numpy=True,
        prompt="Instruct: Classify a Russian Service Desk ticket by category.\nQuery:",
    )
    labels = list(protocol["labels"])
    semantically_filtered = []
    for sample, embedding in zip(rule_filtered, sample_embeddings, strict=True):
        scores = embedding @ label_embeddings.T
        order = np.argsort(scores)[::-1]
        predicted = labels[int(order[0])]
        margin = float(scores[order[0]] - scores[order[1]])
        sample["validation_filters"] = {
            "rules": True,
            "fuzzy_dedup": True,
            "embedding_label": predicted,
            "embedding_margin": round(margin, 6),
        }
        if predicted == sample["category"] and margin >= 0.01:
            semantically_filtered.append((sample, embedding))

    # Greedy embedding dedup is category-local: retain diverse formulations,
    # not dozens of paraphrases that would act as accidental sample weighting.
    filtered = []
    accepted_embeddings: dict[str, list[np.ndarray]] = {}
    for sample, embedding in semantically_filtered:
        prior = accepted_embeddings.setdefault(sample["category"], [])
        if prior and max(float(embedding @ candidate) for candidate in prior) >= 0.96:
            continue
        prior.append(embedding)
        sample["validation_filters"]["embedding_dedup"] = True
        filtered.append(sample)

    prototypes: dict[str, dict[str, Any]] = {}
    for label in labels:
        candidates = [row for row in train_rows if str(row["category"]) == label]
        if candidates:
            prototypes[label] = Counter(
                tuple((key, str(row.get(key) or "")) for key in ("service", "component", "request_type", "criticality", "urgency", "priority", "service_class", "timezone"))
                for row in candidates
            ).most_common(1)[0][0]
    synthetic_rows = []
    for index, sample in enumerate(filtered):
        metadata = dict(prototypes.get(sample["category"], ()))
        synthetic_rows.append(
            {
                **metadata,
                "request_id": f"synthetic-v4-{index}",
                "description": sample["description"],
                "category": sample["category"],
                "is_synthetic": True,
            }
        )

    truth = [str(row["category"]) for row in validation_rows]
    spec = CandidateSpec("category/synthetic-grid/v4", "category", "structured_lr", "combined", "B", {})
    grid: dict[str, Any] = {}
    for ratio_value in (1, 2, 5, 10):
        selected_synthetic = []
        for label in labels:
            candidates = [row for row in synthetic_rows if row["category"] == label]
            selected_synthetic.extend(candidates[: min(len(candidates), support[label] * ratio_value)])
        ratio_grid = {}
        for weight in (0.0, 0.2, 0.3, 0.5, 0.7, 1.0):
            training = [*train_rows, *selected_synthetic]
            weights = np.asarray([1.0] * len(train_rows) + [weight] * len(selected_synthetic))
            estimator = build_estimator(spec)
            estimator.fit(training, [str(row["category"]) for row in training], classifier__sample_weight=weights)
            predicted = [str(value) for value in estimator.predict(validation_rows)]
            ratio_grid[str(weight)] = classification_metrics(truth, predicted, labels=labels)
        grid[f"{ratio_value}x"] = {"synthetic_rows": len(selected_synthetic), "weights": ratio_grid}
    output = {
        "status": "MEASURED_GPU_SYNTHETIC_REAL_ONLY_VALIDATION",
        "generator_model": generator_id,
        "generator_revision": str(generator_info.sha),
        "generated": len(generated),
        "after_rule_filter": len(rule_filtered),
        "after_semantic_filter": len(semantically_filtered),
        "after_independent_embedding_filter": len(filtered),
        "validation_real_only": True,
        "validation_rows": len(validation_rows),
        "grid": grid,
        "samples": filtered,
        "elapsed_timestamp": time.time(),
    }
    destination = PROJECT_ROOT / "artifacts/gpu_research_v4/synthetic/synthetic_experiment.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(destination), "counts": {key: output[key] for key in ("generated", "after_rule_filter", "after_semantic_filter", "after_independent_embedding_filter")}, "grid": grid}, ensure_ascii=False))


if __name__ == "__main__":
    main()
