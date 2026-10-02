from __future__ import annotations

import json
from pathlib import Path

import pytest
from app.ml.v3.candidates import CandidateSpec, build_estimator, fit_group_calibrated_svc, safe_feature_row
from app.ml.v3.experiments import evaluate_candidate
from app.ml.v3.protocol import build_v3_manifest
from app.ml.v3.registry import CandidateRegistry


def test_safe_structured_row_excludes_targets_and_postprocessing_fields() -> None:
    row = {
        "description": "QR не формируется",
        "service": "Получение",
        "component": "QR",
        "request_type": "Инцидент",
        "criticality": "Средняя",
        "urgency": "Средняя",
        "priority": "Высокий",
        "service_class": "Стандарт",
        "timezone": "MSK",
        "category": "LEAK_CATEGORY",
        "final_line": "LEAK_ROUTE",
        "result": "LEAK_RESULT",
        "overdue": 1,
        "actual_duration_seconds": 999,
    }

    safe = safe_feature_row(row)

    assert safe == {
        "description": "QR не формируется",
        "service": "Получение",
        "component": "QR",
        "request_type": "Инцидент",
        "criticality": "Средняя",
        "urgency": "Средняя",
        "priority": "Высокий",
        "service_class": "Стандарт",
        "timezone": "MSK",
    }
    assert "LEAK" not in json.dumps(safe, ensure_ascii=False)


def test_structured_estimator_fits_text_and_metadata_as_separate_representations() -> None:
    rows = [
        {"description": "qr код не работает", "service": "mobile"},
        {"description": "qr ошибка подключения", "service": "mobile"},
        {"description": "qr нет изображения", "service": "mobile"},
        {"description": "zip архив не открывается", "service": "business"},
        {"description": "ошибка импорта zip", "service": "business"},
        {"description": "архив поврежден", "service": "business"},
    ]
    values = [safe_feature_row(row) for row in rows]
    estimator = build_estimator(CandidateSpec("structured_lr", "category", "structured_lr", "combined"))

    estimator.fit(values, ["QR", "QR", "QR", "ZIP", "ZIP", "ZIP"])

    assert estimator.predict([safe_feature_row({"description": "не читается qr", "service": "mobile"})]).tolist() == ["QR"]
    names = estimator.named_steps["features"].get_feature_names_out().tolist()
    assert any("text__" in name for name in names)
    assert any("metadata__" in name for name in names)


def test_candidate_registry_is_idempotent_but_refuses_mutation(tmp_path: Path) -> None:
    path = tmp_path / "registry.json"
    registry = CandidateRegistry(path)
    record = {
        "candidate_id": "category/tfidf_lr/v1",
        "status": "measured",
        "dataset_sha256": "data",
        "split_sha256": "split",
        "metrics": {"cv_macro_f1_mean": 0.7},
    }

    registry.append(record)
    registry.append(record)

    changed = {**record, "metrics": {"cv_macro_f1_mean": 0.9}}
    with pytest.raises(ValueError, match="immutable"):
        registry.append(changed)
    assert json.loads(path.read_text(encoding="utf-8"))["candidates"] == [record]


def test_linear_svc_calibration_uses_disjoint_training_groups() -> None:
    rows = []
    labels = []
    groups = []
    for label, token in (("QR", "qr"), ("ZIP", "zip")):
        for group in range(8):
            for _duplicate in range(2):
                rows.append(safe_feature_row({"description": f"{token} проблема {group}", "service": token}))
                labels.append(label)
                groups.append(f"{label}-{group}")

    model, metadata = fit_group_calibrated_svc(
        CandidateSpec("svc", "category", "calibrated_linear_svc", "combined"),
        rows,
        labels,
        groups,
        seed=17,
    )

    assert set(metadata["fit_groups"]).isdisjoint(metadata["calibration_groups"])
    probabilities = model.predict_proba([safe_feature_row({"description": "qr ошибка", "service": "qr"})])
    assert probabilities.shape == (1, 2)
    assert probabilities.sum() == pytest.approx(1.0)


def test_candidate_evaluation_uses_only_manifest_development_folds() -> None:
    rows = []
    for label, token in (("QR", "qr"), ("ZIP", "zip")):
        for group in range(12):
            for duplicate in range(2):
                rows.append(
                    {
                        "request_id": f"{label}-{group}-{duplicate}",
                        "category": label,
                        "description": f"{token} проблема {group}",
                        "normalized_description": f"{token} проблема {group}",
                        "service": token,
                    }
                )
    manifest = build_v3_manifest(rows, seed=11, repeats=1, n_splits=2)

    result = evaluate_candidate(CandidateSpec("test_lr", "category", "tfidf_lr", "text"), rows, manifest)

    evaluated_ids = {item["request_id"] for item in result["predictions"]}
    assert evaluated_ids.isdisjoint(manifest.holdout_request_ids)
    assert evaluated_ids.isdisjoint(manifest.calibration_request_ids)
    assert result["metrics"]["folds"] == 2
    assert result["metrics"]["cv_macro_f1_mean"] == 1.0


def test_catboost_candidate_uses_native_text_and_categorical_features() -> None:
    pytest.importorskip("catboost")
    rows = []
    labels = []
    for label, service, token in (("QR", "mobile", "qr"), ("ZIP", "business", "zip")):
        for index in range(24):
            rows.append(safe_feature_row({"description": f"{token} уникальная проблема номер {index}", "service": service}))
            labels.append(label)
    estimator = build_estimator(
        CandidateSpec("catboost", "category", "catboost_text", "combined", parameters={"iterations": 30})
    )

    estimator.fit(rows, labels)

    assert estimator.predict([safe_feature_row({"description": "qr уникальная проблема", "service": "mobile"})]).tolist() == ["QR"]
    assert estimator.predict_proba(rows[:1]).shape == (1, 2)
