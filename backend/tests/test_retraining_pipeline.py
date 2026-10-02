from __future__ import annotations

import json
import sqlite3
import subprocess
import sys

import pytest
from app.ml.continuous_learning import build_retraining_dataset, train_challenger
from app.ml.model_registry import ModelRegistry
from app.services.data_service import ensure_database_schema


def _seed_historical(database) -> None:  # noqa: ANN001
    ensure_database_schema(database)
    connection = sqlite3.connect(database)
    rows = [
        (
            "H-1", "2026-01-01", None, "Почта", None, "Категория A", "Инцидент", "Не приходит уведомление",
            "не приходит уведомление", "", "", "Высокий", "", "", "Закрыт", 0, "(1 линия)",
            None, 0, None, 1, 0, 0, 0, 1, "Решено", "{}", "historical-hash", "2026-01-01",
        ),
        (
            "H-2", "2026-01-02", None, "Почта", None, "Категория B", "Инцидент", "Не работает QR код",
            "не работает qr код", "", "", "Средний", "", "", "Закрыт", 0, "(2 линия)",
            None, 0, None, 0, 1, 0, 0, 1, "Решено", "{}", "historical-hash", "2026-01-02",
        ),
    ]
    placeholders = ",".join("?" for _ in rows[0])
    connection.executemany(f"INSERT INTO tickets VALUES ({placeholders})", rows)
    connection.execute(
        """
        INSERT INTO production_feedback (
            incoming_ticket_id, request_id, registration_json,
            model_category, model_category_confidence, model_top3_json,
            model_route, model_route_confidence,
            operator_final_category, operator_final_route,
            category_corrected, route_corrected, accepted_without_change,
            model_version, prediction_timestamp, confirmation_timestamp,
            available_fields_json, missing_fields_json,
            learning_priority, needs_training_review
        ) VALUES (1, 'F-1', ?, 'Категория A', 0.4, '[]', '(1 линия)', 0.5,
                  'Категория B', '(2 линия)', 1, 1, 0, 'v5.2', '2026-09-18', '2026-09-18',
                  '["description"]', '["component"]', 1.0, 1)
        """,
        (json.dumps({"description": "Исправленная оператором проблема", "service": "Почта"}, ensure_ascii=False),),
    )
    connection.commit()
    connection.close()


def test_retraining_dataset_keeps_historical_and_confirmed_feedback_separate(tmp_path) -> None:
    database = tmp_path / "training.sqlite3"
    _seed_historical(database)

    dataset = build_retraining_dataset(database)

    assert dataset["historical_count"] == 2
    assert dataset["feedback_count"] == 1
    assert dataset["rows_total"] == 3
    feedback = next(row for row in dataset["rows"] if row["source"] == "feedback")
    assert feedback["request_id"] == "F-1"
    assert feedback["category"] == "Категория B"
    assert feedback["final_line"] == "(2 линия)"
    assert feedback["group_key"]
    assert all("model_category" not in row["features"] for row in dataset["rows"])


def test_registry_requires_explicit_promotion_and_archives_previous_champion(tmp_path) -> None:
    path = tmp_path / "registry.json"
    registry = ModelRegistry(path)
    registry.register({"version": "v5.2", "status": "champion", "metrics": {"top1": 0.79883}})
    registry.register({"version": "v6.0-001", "status": "challenger", "metrics": {"top1": 0.80}})

    assert registry.champion()["version"] == "v5.2"
    assert registry.get("v6.0-001")["status"] == "challenger"

    registry.promote("v6.0-001")

    assert registry.champion()["version"] == "v6.0-001"
    assert registry.get("v5.2")["status"] == "archived"


def test_registry_rejects_implicit_second_champion(tmp_path) -> None:
    registry = ModelRegistry(tmp_path / "registry.json")
    registry.register({"version": "v5.2", "status": "champion"})
    with pytest.raises(ValueError, match="promote"):
        registry.register({"version": "v6.0", "status": "champion"})


def test_challenger_training_is_group_safe_and_returns_category_and_routing_metrics(tmp_path) -> None:
    rows = []
    for index in range(12):
        category = "Категория A" if index < 6 else "Категория B"
        route = "(1 линия)" if index % 2 == 0 else "(2 линия)"
        rows.append(
            {
                "request_id": f"R-{index}",
                "features": {
                    "description": f"{category} пример номер {index}",
                    "service": "Почта",
                    "component": "",
                    "request_type": "Инцидент",
                    "criticality": "",
                    "urgency": "",
                    "priority": "",
                    "service_class": "",
                    "timezone": "",
                    "registration_date": "",
                    "user": "",
                },
                "category": category,
                "final_line": route,
                "group_key": f"g-{index}",
                "source": "historical",
            }
        )

    result = train_challenger(rows, tmp_path / "challenger.joblib")

    assert (tmp_path / "challenger.joblib").exists()
    assert result["category"]["validation_size"] > 0
    assert 0 <= result["category"]["top1"] <= 1
    assert 0 <= result["category"]["top3"] <= 1
    assert 0 <= result["routing"]["macro_f1"] <= 1
    assert result["split_strategy"] == "GroupShuffleSplit by normalized-description group"


def test_retrain_v6_dry_run_writes_controlled_dataset_report(tmp_path) -> None:
    database = tmp_path / "training.sqlite3"
    _seed_historical(database)
    output_root = tmp_path / "outputs"
    registry = tmp_path / "registry.json"
    command = [
        sys.executable,
        "scripts/retrain_v6.py",
        "--database",
        str(database),
        "--output-root",
        str(output_root),
        "--registry",
        str(registry),
        "--version",
        "v6-test",
        "--dry-run",
    ]
    completed = subprocess.run(command, cwd=".", capture_output=True, text=True, check=False)

    assert completed.returncode == 0, completed.stderr
    report = json.loads((output_root / "v6-test.json").read_text(encoding="utf-8"))
    assert report["mode"] == "dry-run"
    assert report["dataset"]["historical_count"] == 2
    assert report["dataset"]["feedback_count"] == 1
    assert not registry.exists()
