from __future__ import annotations

import json

from app.core.config import DATABASE_PATH, EVALUATION_DIR, MODELS_DIR
from app.ml.runtime import analyze_ticket, load_runtime
from app.ml.training import _model_metadata, make_canonical_split, registration_text
from app.services.data_service import load_rows


def test_model_metadata_is_json_serializable() -> None:
    metadata = _model_metadata("deployment", "word_char_lr", 10, "abc", ["description"], "grouped")
    assert json.loads(json.dumps(metadata))["training_record_count"] == 10


def test_canonical_split_is_group_disjoint_and_has_cv_folds() -> None:
    labels = ["a"] * 12 + ["b"] * 12 + ["c"] * 12
    groups = [f"group-{index // 2}" for index in range(36)]
    split = make_canonical_split(labels, groups, seed=42, outer_splits=3, inner_splits=3)
    development_groups = {groups[index] for index in split.development}
    test_groups = {groups[index] for index in split.test}
    assert development_groups.isdisjoint(test_groups)
    assert len(split.cv_folds) == 3
    for train, validation in split.cv_folds:
        train_groups = {groups[split.development[index]] for index in train}
        validation_groups = {groups[split.development[index]] for index in validation}
        assert train_groups.isdisjoint(validation_groups)


def test_actual_registration_feature_builder_excludes_leakage_sentinels() -> None:
    row = {
        "description": "безопасное описание",
        "service": "safe-service",
        "component": "safe-component",
        "request_type": "safe-type",
        "criticality": "safe-criticality",
        "urgency": "safe-urgency",
        "priority": "safe-priority",
        "service_class": "safe-class",
        "timezone": "safe-zone",
        "category": "LEAK_CATEGORY_123",
        "final_line": "LEAK_FINAL_LINE_456",
        "result": "LEAK_RESULT_WORK_789",
        "overdue": "LEAK_OVERDUE_012",
        "actual_duration_seconds": "LEAK_DURATION_345",
        "raw_json": "LEAK_RAW_678",
    }
    feature = registration_text(row)
    assert "safe-service" in feature
    assert "LEAK_" not in feature


def test_v2_artifacts_record_conflicts_calibration_and_disjoint_splits(prepared_service: None) -> None:
    conflicts = json.loads((EVALUATION_DIR / "label_conflicts.json").read_text(encoding="utf-8"))
    confidence = json.loads((EVALUATION_DIR / "confidence_analysis.json").read_text(encoding="utf-8"))
    split = json.loads((EVALUATION_DIR / "split_metadata.json").read_text(encoding="utf-8"))
    assert "normalized_description_conflicts" in conflicts
    assert {"brier", "ece", "temperature"}.issubset(confidence["calibration"])
    assert split["audit"]["all_disjoint"] is True


def test_deployment_models_use_all_eligible_labeled_rows(prepared_service: None) -> None:
    runtime = load_runtime(MODELS_DIR)
    category_metadata = runtime["category"]["metadata"]
    if category_metadata.get("candidate_id") == "lite-v3-qdistill-customer49-c05-w4":
        assert category_metadata["training_record_count"] == 1997
    else:
        assert category_metadata["training_record_count"] == 1931
    eligible_routing_rows = sum(
        row.get("final_line") in {"(1 линия)", "(2 линия)", "(3 линия)"}
        for row in load_rows(DATABASE_PATH)
    )
    assert runtime["routing"]["metadata"]["training_record_count"] == eligible_routing_rows
    assert category_metadata["candidate_id"] in {
        "lite-v2-tfidf-svc-c035",
        "lite-v3-qdistill-a1-fusion-s1-c035-specialist",
        "lite-v3-qdistill-customer49-c05-w4",
    }
    assert category_metadata["model_family"] in {
        "lite_v2_tfidf_svc",
        "lite_v3_qwen_distilled",
        "lite_v3_qwen_distilled_customer_adapted",
    }


def test_every_model_sidecar_has_reproducibility_metadata(prepared_service: None) -> None:
    required = {
        "role", "model_family", "hyperparameters", "dataset_sha256", "training_record_count",
        "trained_at", "package_versions", "feature_schema", "split_methodology", "model_artifact_sha256",
    }
    for name in ("category", "routing", "retrieval"):
        metadata = json.loads((EVALUATION_DIR / f"{name}_model_metadata.json").read_text(encoding="utf-8"))
        assert required.issubset(metadata), name


def test_lite_v2_low_information_issue_requires_operator_review(prepared_service: None) -> None:
    result = analyze_ticket(
        load_runtime(MODELS_DIR),
        {
            "description": "абракадабра квантовый телепорт океан",
            "service": "", "component": "", "request_type": "", "criticality": "", "urgency": "",
            "priority": "", "service_class": "", "timezone": "",
        },
    )
    assert result["category"]["accepted"] is False
    assert result["category"]["review_required"] is True
    assert result["category"]["confidence"] < result["category"]["threshold"]
    assert result["routing"]["accepted"] is False
    assert result["routing"]["review_required"] is True
    assert len(result["routing"]["alternatives"]) == 3
