from __future__ import annotations

import hashlib
import json
import shutil
import sys
import time
from collections import Counter
from datetime import UTC, datetime
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
from app.ml.v4.embedding_runtime import LazyEmbeddingClassifier  # noqa: E402
from app.ml.v4.evaluation import REGISTRATION_FIELDS, classification_metrics  # noqa: E402
from app.services.data_service import load_rows  # noqa: E402

META_FIELDS = tuple(field for field in REGISTRATION_FIELDS if field != "description")


def _effective_synthetic_weight(sample: dict[str, Any], scale: float) -> float:
    """Scale a recipe while preserving conservative 0-1-real evidence caps."""
    raw = float(sample.get("synthetic_sample_weight", 0.2)) * scale
    if sample.get("evidence_tier") == "SYNTHETIC_ONLY":
        return min(raw, 0.05)
    if sample.get("evidence_tier") == "SINGLE_REAL_ANCHOR":
        return min(raw, 0.10)
    return raw


def _aligned_probabilities(model: Any, values: Any, labels: list[str]) -> np.ndarray:
    raw = np.asarray(model.predict_proba(values))
    positions = {str(label): index for index, label in enumerate(model.classes_)}
    return np.asarray([[row[positions[label]] if label in positions else 0.0 for label in labels] for row in raw])


def _metrics(truth: list[str], probabilities: np.ndarray, labels: list[str]) -> dict[str, Any]:
    predicted = [labels[index] for index in probabilities.argmax(axis=1)]
    return classification_metrics(truth, predicted, labels=labels)


def _training_evidence_bands(metrics: dict[str, Any], real_training_rows: list[dict[str, Any]]) -> dict[str, Any]:
    support = Counter(str(row["category"]) for row in real_training_rows)
    conditions = {
        "zero_real_train": lambda value: value == 0,
        "single_real_anchor": lambda value: value == 1,
        "two_to_five_real": lambda value: 2 <= value <= 5,
    }
    result = {}
    for name, condition in conditions.items():
        labels = [label for label in metrics["per_class"] if condition(support[label])]
        values = [float(metrics["per_class"][label]["f1"]) for label in labels]
        result[name] = {
            "labels": labels,
            "macro_f1": round(float(np.mean(values)), 6) if values else None,
            "real_training_support": {label: support[label] for label in labels},
        }
    return result


def _embedding_matrix(
    embeddings: np.ndarray,
    rows: list[dict[str, Any]],
    indices: list[int],
    *,
    encoder: OneHotEncoder,
    fit: bool,
    scale: float,
) -> sparse.csr_matrix:
    dense = sparse.csr_matrix(embeddings[indices])
    if not scale:
        return dense
    metadata = np.asarray(
        [[safe_feature_row(rows[index])[field] for field in META_FIELDS] for index in indices], dtype=object
    )
    encoded = encoder.fit_transform(metadata) if fit else encoder.transform(metadata)
    return sparse.hstack([dense, encoded * scale], format="csr")


def _audited_synthetic(
    real_rows: list[dict[str, Any]], labels: list[str], recipe: dict[str, Any] | None
) -> list[dict[str, Any]]:
    if not recipe:
        return []
    clean_path = ROOT / "artifacts/gpu_research_v4/synthetic/domain_corpus_clean.jsonl"
    audit = json.loads(
        (ROOT / "artifacts/gpu_research_v4/synthetic/manual_synthetic_audit.json").read_text(encoding="utf-8")
    )
    approved = set(audit["approved_categories"])
    rejected = set(audit.get("rejected_sample_sha256", []))
    clean = [json.loads(line) for line in clean_path.read_text(encoding="utf-8").splitlines() if line]
    support = Counter(str(row["category"]) for row in real_rows)
    fields = META_FIELDS
    prototypes: dict[str, dict[str, str]] = {}
    for label in labels:
        candidates = [safe_feature_row(row) for row in real_rows if str(row["category"]) == label]
        if candidates:
            signature = Counter(tuple((field, str(row.get(field) or "")) for field in fields) for row in candidates)
            prototypes[label] = dict(signature.most_common(1)[0][0])
    ratio = int(recipe["ratio"])
    result = []
    for label in labels:
        candidates = [
            sample
            for sample in clean
            if str(sample["target_category"]) == label
            and label in approved
            and str(sample["sample_sha256"]) not in rejected
        ][: max(1, support[label]) * ratio]
        for sample in candidates:
            result.append(
                {
                    **prototypes.get(label, {}),
                    "request_id": f"synthetic-final-{sample['sample_sha256'][:16]}",
                    "description": str(sample["description"]),
                    "category": label,
                    "sample_sha256": str(sample["sample_sha256"]),
                    "evidence_tier": str(sample.get("evidence_tier", "REAL_ANCHORED")),
                    "synthetic_sample_weight": _effective_synthetic_weight(
                        sample, float(recipe["weight_scale"])
                    ),
                }
            )
    return result


def _evaluate_view(
    family: str,
    view: str,
    rows: list[dict[str, Any]],
    protocol: dict[str, Any],
    labels: list[str],
    synthetic_recipe: dict[str, Any] | None,
) -> tuple[dict[str, Any], float]:
    rows = [row for row in rows if str(row["category"]) in set(labels)]
    by_id = {str(row["request_id"]): index for index, row in enumerate(rows)}
    development = [by_id[item] for item in protocol["development_request_ids"] if item in by_id]
    calibration = [by_id[item] for item in protocol["calibration_request_ids"] if item in by_id]
    holdout = [by_id[item] for item in protocol["holdout_request_ids"] if item in by_id]
    development_rows = [rows[index] for index in development]
    synthetic = _audited_synthetic(development_rows, labels, synthetic_recipe)
    if family == "structured":
        deep = json.loads((ROOT / f"artifacts/gpu_research_v4/deep/structured-{view}.json").read_text())
        parameters = deep["selected"]["parameters"]
        model = build_estimator(
            CandidateSpec("category/final-structured/v4", "category", "structured_lr", "combined", parameters=parameters)
        )
        training_rows = [*development_rows, *synthetic]
        sample_weights = np.asarray(
            [1.0] * len(development_rows)
            + [float(row["synthetic_sample_weight"]) for row in synthetic]
        )
        model.fit(
            [safe_feature_row(row) for row in training_rows],
            [str(row["category"]) for row in training_rows],
            classifier__sample_weight=sample_weights,
        )
        calibration_values = [safe_feature_row(rows[index]) for index in calibration]
        holdout_values = [safe_feature_row(rows[index]) for index in holdout]
        calibration_probabilities = _aligned_probabilities(model, calibration_values, labels)
        started = time.perf_counter()
        holdout_probabilities = _aligned_probabilities(model, holdout_values, labels)
        latency_ms = 1000 * (time.perf_counter() - started) / max(1, len(holdout))
    else:
        slug = "bge-m3" if family == "bge-m3" else "qwen3-embedding-4b"
        deep = json.loads((ROOT / f"artifacts/gpu_research_v4/deep/{slug}-{view}.json").read_text())
        parameters = deep["selected"]["parameters"]
        archive = np.load(ROOT / f"artifacts/gpu_research_v4/deep/{slug}-{view}.embeddings.npz")
        embedding_by_id = {str(item): index for index, item in enumerate(archive["request_ids"])}
        embeddings = np.asarray([archive["embeddings"][embedding_by_id[str(row["request_id"])]] for row in rows])
        metadata_encoder = OneHotEncoder(handle_unknown="ignore", min_frequency=2)
        scale = float(parameters["metadata_scale"])
        train_matrix = _embedding_matrix(
            embeddings, rows, development, encoder=metadata_encoder, fit=True, scale=scale
        )
        calibration_matrix = _embedding_matrix(
            embeddings, rows, calibration, encoder=metadata_encoder, fit=False, scale=scale
        )
        holdout_matrix = _embedding_matrix(
            embeddings, rows, holdout, encoder=metadata_encoder, fit=False, scale=scale
        )
        model = LogisticRegression(
            C=float(parameters["C"]),
            max_iter=3000,
            class_weight=parameters["class_weight"],
            random_state=20260921,
        )
        train_targets = [str(rows[index]["category"]) for index in development]
        train_weights = [1.0] * len(development)
        if synthetic:
            synthetic_archive = np.load(
                ROOT / f"artifacts/gpu_research_v4/synthetic/{slug}-clean.embeddings.npz"
            )
            synthetic_by_hash = {
                str(value): index for index, value in enumerate(synthetic_archive["sample_sha256"])
            }
            synthetic_indices = [synthetic_by_hash[row["sample_sha256"]] for row in synthetic]
            synthetic_embeddings = np.asarray(synthetic_archive["embeddings"])[synthetic_indices]
            synthetic_metadata = np.asarray(
                [[safe_feature_row(row)[field] for field in META_FIELDS] for row in synthetic], dtype=object
            )
            synthetic_matrix: Any = sparse.csr_matrix(synthetic_embeddings)
            if scale:
                synthetic_matrix = sparse.hstack(
                    [synthetic_matrix, metadata_encoder.transform(synthetic_metadata) * scale], format="csr"
                )
            train_matrix = sparse.vstack([train_matrix, synthetic_matrix], format="csr")
            train_targets.extend(str(row["category"]) for row in synthetic)
            train_weights.extend(float(row["synthetic_sample_weight"]) for row in synthetic)
        model.fit(train_matrix, train_targets, sample_weight=np.asarray(train_weights))
        calibration_probabilities = _aligned_probabilities(model, calibration_matrix, labels)
        started = time.perf_counter()
        holdout_probabilities = _aligned_probabilities(model, holdout_matrix, labels)
        latency_ms = 1000 * (time.perf_counter() - started) / max(1, len(holdout)) + float(deep["encode_ms_per_ticket"])
    calibration_truth = [str(rows[index]["category"]) for index in calibration]
    calibration_prediction = np.asarray(labels)[calibration_probabilities.argmax(axis=1)]
    calibration_confidence = calibration_probabilities.max(axis=1)
    correct = calibration_prediction == np.asarray(calibration_truth)
    threshold = float(np.quantile(calibration_confidence[correct], 0.05)) if correct.any() else 0.0
    holdout_truth = [str(rows[index]["category"]) for index in holdout]
    holdout_prediction = np.asarray(labels)[holdout_probabilities.argmax(axis=1)]
    holdout_confidence = holdout_probabilities.max(axis=1)
    holdout_metrics = _metrics(holdout_truth, holdout_probabilities, labels)
    return (
        {
            "development_rows": len(development),
            "calibration_rows": len(calibration),
            "holdout_rows": len(holdout),
            "taxonomy_labels": len(labels),
            "synthetic_training_rows": len(synthetic),
            "synthetic_recipe": synthetic_recipe,
            "metrics": holdout_metrics,
            "training_evidence_bands": _training_evidence_bands(holdout_metrics, development_rows),
            "known_false_unknown_rate": round(float((holdout_confidence < threshold).mean()), 6),
            "accepted_accuracy": round(
                float((holdout_prediction[holdout_confidence >= threshold] == np.asarray(holdout_truth)[holdout_confidence >= threshold]).mean()),
                6,
            ) if (holdout_confidence >= threshold).any() else 0.0,
            "latency_ms_per_ticket_including_encoder": round(latency_ms, 4),
        },
        threshold,
    )


def main() -> None:
    selection_path = ROOT / "artifacts/gpu_research_v4/final/selection.json"
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if selection["status"] != "FROZEN_BEFORE_SEALED_HOLDOUT":
        raise RuntimeError("finalist selection must be frozen before accessing holdout")
    family = str(selection["production_winner"]["family"])
    if family not in {"structured", "bge-m3", "qwen3-embedding-4b"}:
        raise RuntimeError(f"production export does not support {family}")
    pointer = json.loads((ROOT / "artifacts/gpu_research_v4/protocol.json").read_text())
    protocol = json.loads((ROOT / pointer["protocol_path"]).read_text())
    rows = [row for row in load_rows(DATABASE_PATH) if str(row.get("description") or "").strip()]
    counts = Counter(str(row["category"]) for row in rows)
    top15 = [label for label, _ in counts.most_common(15)]
    full43 = sorted(counts)
    synthetic_recipe = selection["production_winner"].get("synthetic_recipe")
    top_report, _ = _evaluate_view(family, "top15", rows, protocol, top15, synthetic_recipe)
    full_report, threshold = _evaluate_view(family, "full43", rows, protocol, full43, synthetic_recipe)

    models_dir = ROOT / "models"
    v4_dir = models_dir / "v4"
    v4_dir.mkdir(parents=True, exist_ok=True)
    if family == "structured":
        parameters = selection["production_winner"]["parameters"]
        deployment = build_estimator(
            CandidateSpec("category/structured-deployment/v4", "category", "structured_lr", "combined", parameters=parameters)
        )
        synthetic = _audited_synthetic(rows, full43, synthetic_recipe)
        deployment_rows = [*rows, *synthetic]
        deployment.fit(
            [safe_feature_row(row) for row in deployment_rows],
            [str(row["category"]) for row in deployment_rows],
            classifier__sample_weight=np.asarray(
                [1.0] * len(rows) + [float(row["synthetic_sample_weight"]) for row in synthetic]
            ),
        )
        pipeline: Any = deployment
        encoder_metadata: dict[str, Any] = {"family": "structured_word_char_tfidf_lr"}
    else:
        slug = "bge-m3" if family == "bge-m3" else "qwen3-embedding-4b"
        deployment_artifact = joblib.load(ROOT / f"artifacts/gpu_research_v4/deep/{slug}-full43.joblib")
        synthetic = _audited_synthetic(rows, full43, synthetic_recipe)
        if synthetic:
            real_archive = np.load(ROOT / f"artifacts/gpu_research_v4/deep/{slug}-full43.embeddings.npz")
            real_by_id = {str(value): index for index, value in enumerate(real_archive["request_ids"])}
            real_embeddings = np.asarray(
                [real_archive["embeddings"][real_by_id[str(row["request_id"])]] for row in rows]
            )
            synthetic_archive = np.load(
                ROOT / f"artifacts/gpu_research_v4/synthetic/{slug}-clean.embeddings.npz"
            )
            synthetic_by_hash = {
                str(value): index for index, value in enumerate(synthetic_archive["sample_sha256"])
            }
            synthetic_embeddings = np.asarray(
                [synthetic_archive["embeddings"][synthetic_by_hash[row["sample_sha256"]]] for row in synthetic]
            )
            metadata_encoder = OneHotEncoder(handle_unknown="ignore", min_frequency=2)
            real_metadata = np.asarray(
                [[safe_feature_row(row)[field] for field in META_FIELDS] for row in rows], dtype=object
            )
            synthetic_metadata = np.asarray(
                [[safe_feature_row(row)[field] for field in META_FIELDS] for row in synthetic], dtype=object
            )
            scale = float(deployment_artifact["metadata_scale"])
            real_matrix: Any = sparse.csr_matrix(real_embeddings)
            synthetic_matrix = sparse.csr_matrix(synthetic_embeddings)
            if scale:
                real_meta_matrix = metadata_encoder.fit_transform(real_metadata) * scale
                synthetic_meta_matrix = metadata_encoder.transform(synthetic_metadata) * scale
                real_matrix = sparse.hstack([real_matrix, real_meta_matrix], format="csr")
                synthetic_matrix = sparse.hstack([synthetic_matrix, synthetic_meta_matrix], format="csr")
            classifier = LogisticRegression(
                C=float(deployment_artifact["selected_parameters"]["C"]),
                max_iter=3000,
                class_weight=deployment_artifact["selected_parameters"]["class_weight"],
                random_state=20260924,
            )
            classifier.fit(
                sparse.vstack([real_matrix, synthetic_matrix], format="csr"),
                [str(row["category"]) for row in rows] + [str(row["category"]) for row in synthetic],
                sample_weight=np.asarray(
                    [1.0] * len(rows) + [float(row["synthetic_sample_weight"]) for row in synthetic]
                ),
            )
            deployment_artifact["metadata_encoder"] = metadata_encoder
            deployment_artifact["classifier"] = classifier
        snapshot = (
            Path("/workspace/hf-cache")
            / f"models--{deployment_artifact['model_id'].replace('/', '--')}"
            / "snapshots"
            / deployment_artifact["revision"]
        )
        encoder_dir = v4_dir / "encoder"
        if snapshot.exists():
            shutil.copytree(snapshot, encoder_dir, dirs_exist_ok=True, symlinks=False)
            deployment_artifact["encoder_path"] = "models/v4/encoder"
        pipeline = LazyEmbeddingClassifier(deployment_artifact, cache_dir=Path("/workspace/hf-cache"))
        encoder_metadata = {
            "family": "frozen_embedding_plus_structured_metadata_lr",
            "model_id": deployment_artifact["model_id"],
            "revision": deployment_artifact["revision"],
            "license": deployment_artifact["license"],
            "encoder_path": deployment_artifact.get("encoder_path"),
        }
    trained_at = datetime.now(UTC).isoformat()
    bundle = {
        "pipeline": pipeline,
        "explanation_pipeline": None,
        "input_contract": "registration_mapping",
        "top15": top15,
        "threshold": threshold,
        "margin_threshold": 0.0,
        "metadata": {
            "candidate_id": selection["production_winner"]["full43_candidate_id"],
            **encoder_metadata,
            "dataset_sha256": protocol["dataset_sha256"],
            "split_sha256": protocol["split_sha256"],
            "training_record_count": len(rows),
            "synthetic_training_record_count": len(synthetic),
            "synthetic_recipe": synthetic_recipe,
            "trained_at": trained_at,
        },
    }
    model_path = v4_dir / "category.joblib"
    joblib.dump(bundle, model_path)
    artifact_sha = hashlib.sha256(model_path.read_bytes()).hexdigest()
    champion = {
        **bundle["metadata"],
        "artifact": str(model_path.relative_to(ROOT)),
        "artifact_sha256": artifact_sha,
        "holdout": {"top15": top_report, "full43": full_report},
    }
    registry = {
        "ACTIVE_CHAMPION": champion,
        "PREVIOUS_CHAMPION": {"candidate_id": "category/tuned_word_char_lr/v3", "artifact": "models/category.joblib"},
        "CHALLENGERS": [item for item in selection["candidates"] if item["family"] != family],
    }
    (v4_dir / "registry.json").write_text(json.dumps(registry, ensure_ascii=False, indent=2) + "\n")
    policy = {"unknown_label": "UNKNOWN_NEW_ISSUE", "known_threshold": threshold, "champion": champion}
    (models_dir / "v4_runtime.json").write_text(json.dumps(policy, ensure_ascii=False, indent=2) + "\n")
    result = {
        "candidate_id": champion["candidate_id"],
        "status": "FINAL_SEALED_EVALUATION_COMPLETE",
        "selection_artifact": str(selection_path.relative_to(ROOT)),
        "selection_frozen_before_holdout": True,
        "sealed_holdout_accessed_once": True,
        "real_only_evaluation": True,
        "metrics": {"top15": top_report, "full43": full_report},
        "unknown_threshold_from_real_calibration": threshold,
        "registry": registry,
        "runtime_verification": {},
    }
    destination = ROOT / "artifacts/gpu_research_v4/final/final-evaluation.json"
    destination.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"output": str(destination), "candidate": champion["candidate_id"], "metrics": result["metrics"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
