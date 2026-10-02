from __future__ import annotations

import hashlib
import json
import os
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
ART = ROOT / "artifacts/gpu_research_v4/synthetic"


def _hash(text: str) -> str:
    return hashlib.sha256(" ".join(text.lower().replace("ё", "е").split()).encode("utf-8")).hexdigest()


def _wait(path: Path) -> None:
    deadline = time.monotonic() + 12 * 60 * 60
    while not path.exists() and time.monotonic() < deadline:
        time.sleep(15)
    if not path.exists():
        raise RuntimeError(f"timed out waiting for {path}")


def main() -> None:
    import torch
    from sentence_transformers import SentenceTransformer

    raw_path = ART / "domain_corpus_raw.jsonl"
    qwen_path = ART / "synthetic_experiment.json"
    _wait(raw_path)
    _wait(qwen_path)
    raw = [json.loads(line) for line in raw_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    qwen_source = json.loads(qwen_path.read_text(encoding="utf-8"))
    combined = list(raw)
    for sample in qwen_source["samples"]:
        text = str(sample["description"])
        category = str(sample["category"])
        combined.append(
            {
                "sample_sha256": _hash(text),
                "description": text,
                "target_category": category,
                "generation_method": "local_qwen_generation_v4",
                "prompt_template_version": sample.get("prompt_version", "v4.1"),
                "generator_model": sample.get("generator_model"),
                "generator_revision": sample.get("generator_revision"),
                "seed": sample.get("seed"),
                "source_definition": sample.get("source_category"),
                "generation_prompt": (
                    "Generate realistic anonymized Russian Service Desk tickets for category "
                    f"{category}; use the category definition and real TRAIN examples only for style; "
                    "return JSON; include typos, abbreviations and varied lengths; do not copy examples."
                ),
                "hard_negative_confusion_target": None,
                "quality_filters_passed": list(sample.get("validation_filters", {}).keys()),
                "synthetic_sample_weight": 0.2,
            }
        )

    definitions = yaml.safe_load((ROOT / "data/category_definitions_v4.yaml").read_text(encoding="utf-8"))
    labels = [str(item["category"]) for item in definitions]
    definition_by_label = {str(item["category"]): item for item in definitions}
    profiles = json.loads((ART / "real_train_category_profiles.json").read_text(encoding="utf-8"))["profiles"]
    label_texts = [
        " ".join(
            [
                str(item.get("meaning") or item["definition"]),
                "Диагностические признаки:",
                "; ".join(item.get("diagnostic_terms", [])),
                "Не относится:",
                "; ".join(item.get("does_not_belong", [])),
            ]
        )
        for item in definitions
    ]
    model = SentenceTransformer(
        "Qwen/Qwen3-Embedding-0.6B",
        cache_folder=os.environ.get("HF_HOME"),
        device="cpu",
        model_kwargs={"torch_dtype": torch.float32},
    )
    instruction = "Instruct: Classify a Russian Service Desk ticket by its operational issue category.\nQuery:"
    label_embeddings = model.encode(label_texts, normalize_embeddings=True, batch_size=16, convert_to_numpy=True)
    sample_embeddings = model.encode(
        [str(sample["description"]) for sample in combined],
        prompt=instruction,
        normalize_embeddings=True,
        batch_size=16,
        convert_to_numpy=True,
        show_progress_bar=True,
    )

    # Real TRAIN only contamination comparison.
    pointer = json.loads((ROOT / "artifacts/gpu_research_v4/protocol.json").read_text(encoding="utf-8"))
    protocol = json.loads((ROOT / pointer["protocol_path"]).read_text(encoding="utf-8"))
    fold = next(item for item in protocol["folds"] if int(item["repeat"]) == 0 and int(item["fold"]) == 0)
    import sqlite3

    connection = sqlite3.connect(ROOT / "data/processed/posttech.db")
    connection.row_factory = sqlite3.Row
    by_id = {str(row["request_id"]): dict(row) for row in connection.execute("SELECT * FROM tickets")}
    real_texts = [str(by_id[str(item)]["description"]) for item in fold["train_request_ids"]]
    real_embeddings = model.encode(
        real_texts,
        prompt=instruction,
        normalize_embeddings=True,
        batch_size=16,
        convert_to_numpy=True,
        show_progress_bar=True,
    )

    kept: list[dict[str, Any]] = []
    kept_embeddings: dict[str, list[np.ndarray]] = defaultdict(list)
    rejected = Counter()
    for sample, embedding in zip(combined, sample_embeddings, strict=True):
        label = str(sample["target_category"])
        if label not in definition_by_label:
            rejected["unknown_label"] += 1
            continue
        token_count = len(str(sample["description"]).split())
        if token_count < 4 or token_count > 500:
            rejected["style_length"] += 1
            continue
        scores = embedding @ label_embeddings.T
        target_index = labels.index(label)
        order = np.argsort(scores)[::-1]
        rank = int(np.where(order == target_index)[0][0]) + 1
        conflict = sample.get("hard_negative_confusion_target")
        conflict_score = float(scores[labels.index(conflict)]) if conflict in labels else float(scores[order[1]])
        margin = float(scores[target_index] - conflict_score)
        hard_case = bool(conflict)
        if rank > (3 if hard_case else 2) or margin < (-0.025 if hard_case else -0.01):
            rejected["category_semantic_mismatch"] += 1
            continue
        if float(np.max(embedding @ real_embeddings.T)) >= 0.965:
            rejected["embedding_real_contamination"] += 1
            continue
        prior = kept_embeddings[label]
        if prior and max(float(embedding @ previous) for previous in prior) >= 0.94:
            rejected["embedding_near_duplicate"] += 1
            continue
        prior.append(embedding)
        record = dict(sample)
        record.setdefault(
            "generation_prompt",
            (
                f"Generate a realistic Russian Service Desk ticket for {label}; require one category-specific "
                f"diagnostic signal; imitate frozen real-TRAIN style; contrast with "
                f"{record.get('hard_negative_confusion_target') or 'no explicit neighbor'}; do not copy real text."
            ),
        )
        record["source_definition_snapshot"] = definition_by_label[label]
        real_support = int(profiles[label]["real_train_support"])
        if real_support == 0:
            record["evidence_tier"] = "SYNTHETIC_ONLY"
            record["synthetic_sample_weight"] = min(float(record.get("synthetic_sample_weight", 0.2)), 0.05)
        elif real_support == 1:
            record["evidence_tier"] = "SINGLE_REAL_ANCHOR"
            record["synthetic_sample_weight"] = min(float(record.get("synthetic_sample_weight", 0.2)), 0.10)
        else:
            record["evidence_tier"] = "REAL_ANCHORED"
        record["quality_filters_passed"] = sorted(
            set(record.get("quality_filters_passed", []))
            | {"style_check", "category_semantic_check", "confusion_margin_check", "embedding_real_contamination", "embedding_diversity"}
        )
        record["quality_scores"] = {
            "target_rank": rank,
            "target_similarity": round(float(scores[target_index]), 6),
            "conflict_margin": round(margin, 6),
            "max_real_train_similarity": round(float(np.max(embedding @ real_embeddings.T)), 6),
        }
        record["real_train_style"] = profiles[label]["style"]
        kept.append(record)

    clean_path = ART / "domain_corpus_clean.jsonl"
    clean_path.write_text("".join(json.dumps(sample, ensure_ascii=False) + "\n" for sample in kept), encoding="utf-8")
    by_category = Counter(str(sample["target_category"]) for sample in kept)
    manifest = {
        "status": "CLEAN_AWAITING_REPRESENTATIVE_SEMANTIC_AUDIT",
        "rows_before_qc": len(combined),
        "rows_after_qc": len(kept),
        "by_generation_method": dict(Counter(str(sample["generation_method"]) for sample in kept)),
        "by_category": dict(by_category),
        "rejections": dict(rejected),
        "embedding_qc_model": "Qwen/Qwen3-Embedding-0.6B",
        "real_only_seed_scope": "frozen_v4_repeat0_fold0_train",
        "validation_calibration_holdout_used": False,
        "source_files": [str(raw_path.relative_to(ROOT)), str(qwen_path.relative_to(ROOT))],
        "zero_real_train_policy": "SYNTHETIC_ONLY; max sample weight 0.05; taxonomy/official sources only",
        "single_real_train_policy": "SINGLE_REAL_ANCHOR; max sample weight 0.10; diversity not paraphrase",
    }
    (ART / "domain_corpus_clean_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    audit_classes = []
    for label in labels:
        candidates = [sample for sample in kept if sample["target_category"] == label]
        candidates.sort(key=lambda item: float(item["quality_scores"]["conflict_margin"]))
        positions = sorted({0, len(candidates) // 2, len(candidates) - 1}) if candidates else []
        audit_classes.append(
            {
                "category": label,
                "meaning": definition_by_label[label].get("meaning"),
                "real_train_support": profiles[label]["real_train_support"],
                "confusable_categories": profiles[label]["empirical_confusion_categories"],
                "clean_samples": len(candidates),
                "representative_samples": [candidates[position] for position in positions],
                "semantic_decision": "PENDING",
                "review_notes": "",
            }
        )
    audit = {"status": "PENDING_REPRESENTATIVE_SEMANTIC_AUDIT", "manifest": manifest, "classes": audit_classes}
    (ART / "domain_corpus_audit_pack.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()
