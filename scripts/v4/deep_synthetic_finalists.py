from __future__ import annotations

import hashlib
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

from app.core.config import DATABASE_PATH  # noqa: E402
from app.ml.v3.candidates import CandidateSpec, build_estimator, safe_feature_row  # noqa: E402
from app.ml.v4.evaluation import REGISTRATION_FIELDS, classification_metrics  # noqa: E402
from app.services.data_service import load_rows  # noqa: E402

META_FIELDS = tuple(field for field in REGISTRATION_FIELDS if field != "description")


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _choose(samples: list[dict[str, Any]], support: Counter[str], ratio: int) -> list[dict[str, Any]]:
    result = []
    for label in sorted(support):
        candidates = [sample for sample in samples if str(sample["category"]) == label]
        result.extend(candidates[: min(len(candidates), max(1, support[label]) * ratio)])
    return result


def _effective_weight(sample: dict[str, Any], scale: float) -> float:
    """Scale synthetic evidence without violating the 0-1-real class caps."""
    raw = scale * float(sample["synthetic_sample_weight"])
    if sample.get("evidence_tier") == "SYNTHETIC_ONLY":
        return min(raw, 0.05)
    if sample.get("evidence_tier") == "SINGLE_REAL_ANCHOR":
        return min(raw, 0.10)
    return raw


def _score(metrics: dict[str, Any]) -> float:
    rare = float(metrics["support_bands"]["1-5"]["macro_f1"] or 0.0)
    return float(metrics["macro_f1"]) + 0.25 * float(metrics["balanced_accuracy"]) + 0.25 * rare + 0.1 * float(metrics["worst_class_f1"])


def _survives(baseline: dict[str, Any], best: dict[str, Any]) -> bool:
    base = baseline["metrics"]
    candidate = best["metrics"]
    base_rare = float(base["support_bands"]["1-5"]["macro_f1"] or 0.0)
    candidate_rare = float(candidate["support_bands"]["1-5"]["macro_f1"] or 0.0)
    return (
        float(candidate["macro_f1"]) > float(base["macro_f1"])
        and candidate_rare >= base_rare - 0.01
        and float(candidate["worst_class_f1"]) >= float(base["worst_class_f1"])
    )


def _evidence_band_metrics(metrics: dict[str, Any], support: Counter[str]) -> dict[str, Any]:
    bands = {
        "zero_real_train": [label for label, value in support.items() if value == 0],
        "single_real_anchor": [label for label, value in support.items() if value == 1],
        "two_to_five_real": [label for label, value in support.items() if 2 <= value <= 5],
    }
    # Counter does not retain labels absent from train; recover them from the per-class table.
    bands["zero_real_train"] = [
        label for label in metrics["per_class"] if support[label] == 0
    ]
    result = {}
    for name, labels in bands.items():
        values = [float(metrics["per_class"][label]["f1"]) for label in labels if label in metrics["per_class"]]
        result[name] = {
            "labels": labels,
            "macro_f1": round(float(np.mean(values)), 6) if values else None,
            "note": "reported separately; synthetic rows are not genuine supervised evidence",
        }
    return result


def _metadata_prototypes(train: list[dict[str, Any]], labels: list[str]) -> dict[str, dict[str, str]]:
    prototypes = {}
    for label in labels:
        rows = [safe_feature_row(row) for row in train if str(row["category"]) == label]
        if rows:
            signature = Counter(tuple((field, str(row.get(field) or "")) for field in META_FIELDS) for row in rows)
            prototypes[label] = dict(signature.most_common(1)[0][0])
    return prototypes


def main() -> None:
    import torch
    from sentence_transformers import SentenceTransformer

    pointer = json.loads((ROOT / "artifacts/gpu_research_v4/protocol.json").read_text(encoding="utf-8"))
    protocol = json.loads((ROOT / pointer["protocol_path"]).read_text(encoding="utf-8"))
    fold = next(item for item in protocol["folds"] if int(item["repeat"]) == 0 and int(item["fold"]) == 0)
    rows = load_rows(DATABASE_PATH)
    by_id = {str(row["request_id"]): row for row in rows}
    train = [by_id[item] for item in fold["train_request_ids"]]
    validation = [by_id[item] for item in fold["validation_request_ids"]]
    labels = sorted(str(label) for label in protocol["labels"])
    support = Counter(str(row["category"]) for row in train)
    for label in labels:
        support.setdefault(label, 0)
    truth = [str(row["category"]) for row in validation]
    prototypes = _metadata_prototypes(train, labels)

    clean_path = ROOT / "artifacts/gpu_research_v4/synthetic/domain_corpus_clean.jsonl"
    audit_path = ROOT / "artifacts/gpu_research_v4/synthetic/manual_synthetic_audit.json"
    clean = [json.loads(line) for line in clean_path.read_text(encoding="utf-8").splitlines() if line]
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    approved = set(audit["approved_categories"])
    rejected = set(audit.get("rejected_sample_sha256", []))
    samples = [sample for sample in clean if sample["target_category"] in approved and sample["sample_sha256"] not in rejected]
    synthetic_rows = [
        {
            **prototypes.get(str(sample["target_category"]), {}),
            "request_id": f"synthetic-finalist-{index}",
            "description": str(sample["description"]),
            "category": str(sample["target_category"]),
            "synthetic_sample_weight": float(sample.get("synthetic_sample_weight", 0.2)),
            "evidence_tier": str(sample.get("evidence_tier", "REAL_ANCHORED")),
            "sample_sha256": str(sample["sample_sha256"]),
        }
        for index, sample in enumerate(samples)
    ]
    families: dict[str, Any] = {}

    structured_parameters = json.loads((ROOT / "artifacts/gpu_research_v4/deep/structured-full43.json").read_text())["selected"]["parameters"]
    structured_grid = []
    for ratio in (0, 10, 25, 50, 100):
        chosen = [] if ratio == 0 else _choose(synthetic_rows, support, ratio)
        for scale in ((0.0,) if ratio == 0 else (0.5, 1.0, 1.5, 2.0)):
            model = build_estimator(CandidateSpec("synthetic-finalist", "category", "structured_lr", "combined", parameters=structured_parameters))
            training = [*train, *chosen]
            weights = np.asarray([1.0] * len(train) + [_effective_weight(row, scale) for row in chosen])
            model.fit(training, [str(row["category"]) for row in training], classifier__sample_weight=weights)
            predicted = [str(value) for value in model.predict(validation)]
            metrics = classification_metrics(truth, predicted, labels=labels)
            structured_grid.append({"ratio": ratio, "weight_scale": scale, "synthetic_rows": len(chosen), "objective": round(_score(metrics), 8), "metrics": metrics})
    structured_baseline = structured_grid[0]
    structured_best = max(structured_grid[1:], key=lambda item: item["objective"])
    families["structured"] = {"baseline": structured_baseline, "best": structured_best, "accepted": _survives(structured_baseline, structured_best), "grid": structured_grid}

    for family, slug, model_id in (
        ("bge-m3", "bge-m3", "BAAI/bge-m3"),
        ("qwen3-embedding-4b", "qwen3-embedding-4b", "Qwen/Qwen3-Embedding-4B"),
    ):
        deep = json.loads((ROOT / f"artifacts/gpu_research_v4/deep/{slug}-full43.json").read_text())
        artifact = joblib.load(ROOT / f"artifacts/gpu_research_v4/deep/{slug}-full43.joblib")
        archive = np.load(ROOT / f"artifacts/gpu_research_v4/deep/{slug}-full43.embeddings.npz")
        embedding_by_id = {str(value): index for index, value in enumerate(archive["request_ids"])}
        real_embeddings = np.asarray(archive["embeddings"], dtype=np.float32)
        train_embeddings = np.asarray([real_embeddings[embedding_by_id[str(row["request_id"])]] for row in train])
        validation_embeddings = np.asarray([real_embeddings[embedding_by_id[str(row["request_id"])]] for row in validation])
        encoder = SentenceTransformer(
            model_id,
            revision=artifact["revision"],
            cache_folder=os.environ.get("HF_HOME"),
            trust_remote_code=False,
            model_kwargs={"torch_dtype": torch.bfloat16},
        )
        encode_kwargs: dict[str, Any] = {"batch_size": 8, "normalize_embeddings": True, "convert_to_numpy": True, "show_progress_bar": True}
        if artifact.get("instruction"):
            encode_kwargs["prompt"] = f"Instruct: {artifact['instruction']}\nQuery:"
        synthetic_embeddings = np.asarray(encoder.encode([row["description"] for row in synthetic_rows], **encode_kwargs), dtype=np.float32)
        del encoder
        torch.cuda.empty_cache()
        np.savez_compressed(
            ROOT / f"artifacts/gpu_research_v4/synthetic/{slug}-clean.embeddings.npz",
            sample_sha256=np.asarray([row["sample_sha256"] for row in synthetic_rows], dtype=str),
            embeddings=synthetic_embeddings,
        )
        parameters = deep["selected"]["parameters"]
        metadata_scale = float(parameters["metadata_scale"])
        meta_encoder = OneHotEncoder(handle_unknown="ignore", min_frequency=2)
        train_meta = np.asarray([[safe_feature_row(row)[field] for field in META_FIELDS] for row in train], dtype=object)
        val_meta = np.asarray([[safe_feature_row(row)[field] for field in META_FIELDS] for row in validation], dtype=object)
        synthetic_meta = np.asarray([[safe_feature_row(row)[field] for field in META_FIELDS] for row in synthetic_rows], dtype=object)
        train_meta_matrix = meta_encoder.fit_transform(train_meta) * metadata_scale
        val_meta_matrix = meta_encoder.transform(val_meta) * metadata_scale
        synthetic_meta_matrix = meta_encoder.transform(synthetic_meta) * metadata_scale
        x_train = sparse.hstack([sparse.csr_matrix(train_embeddings), train_meta_matrix], format="csr") if metadata_scale else sparse.csr_matrix(train_embeddings)
        x_val = sparse.hstack([sparse.csr_matrix(validation_embeddings), val_meta_matrix], format="csr") if metadata_scale else sparse.csr_matrix(validation_embeddings)
        family_grid = []
        for ratio in (0, 10, 25, 50, 100):
            chosen_rows = [] if ratio == 0 else _choose(synthetic_rows, support, ratio)
            chosen_hashes = {row["sample_sha256"] for row in chosen_rows}
            chosen_indices = [index for index, row in enumerate(synthetic_rows) if row["sample_sha256"] in chosen_hashes]
            chosen_matrix = sparse.csr_matrix(synthetic_embeddings[chosen_indices])
            if metadata_scale and chosen_indices:
                chosen_matrix = sparse.hstack([chosen_matrix, synthetic_meta_matrix[chosen_indices]], format="csr")
            for scale in ((0.0,) if ratio == 0 else (0.5, 1.0, 1.5, 2.0)):
                matrix = x_train if not chosen_indices else sparse.vstack([x_train, chosen_matrix], format="csr")
                targets = [str(row["category"]) for row in train] + [str(synthetic_rows[index]["category"]) for index in chosen_indices]
                weights = np.asarray(
                    [1.0] * len(train)
                    + [_effective_weight(synthetic_rows[index], scale) for index in chosen_indices]
                )
                classifier = LogisticRegression(C=float(parameters["C"]), max_iter=3000, class_weight=parameters["class_weight"], random_state=20260923)
                classifier.fit(matrix, targets, sample_weight=weights)
                predicted = [str(value) for value in classifier.predict(x_val)]
                metrics = classification_metrics(truth, predicted, labels=labels)
                family_grid.append({"ratio": ratio, "weight_scale": scale, "synthetic_rows": len(chosen_indices), "objective": round(_score(metrics), 8), "metrics": metrics})
        baseline = family_grid[0]
        best = max(family_grid[1:], key=lambda item: item["objective"])
        families[family] = {"baseline": baseline, "best": best, "accepted": _survives(baseline, best), "grid": family_grid}

    result = {
        "status": "MEASURED_ON_REAL_VALIDATION_ONLY",
        "corpus": str(clean_path.relative_to(ROOT)),
        "corpus_sha256": _hash_file(clean_path),
        "manual_audit": str(audit_path.relative_to(ROOT)),
        "train_synthetic_only": True,
        "validation_real_only": True,
        "calibration_or_holdout_accessed": False,
        "families": families,
        "training_evidence_bands": {
            family: {
                "baseline": _evidence_band_metrics(values["baseline"]["metrics"], support),
                "best": _evidence_band_metrics(values["best"]["metrics"], support),
            }
            for family, values in families.items()
        },
        "zero_real_train_policy": {
            "sample_mark": "SYNTHETIC_ONLY",
            "max_individual_weight": 0.05,
            "definition_sources": "known taxonomy, official/public documentation, training-safe information only",
            "complementary_methods": ["label-semantic/NLI", "prototype/nearest-class", "contrastive screening"],
        },
    }
    destination = ROOT / "artifacts/gpu_research_v4/deep/synthetic-finalists.json"
    destination.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(destination), "families": {name: {"accepted": value["accepted"], "best": value["best"]} for name, value in families.items()}}, ensure_ascii=False))


if __name__ == "__main__":
    main()
