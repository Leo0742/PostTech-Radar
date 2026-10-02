from __future__ import annotations

import pytest
from app.core.config import DATABASE_PATH, MODELS_DIR, PROJECT_ROOT
from app.ml.runtime import analyze_ticket, load_runtime
from app.services.analytics_service import AnalyticsFilters, analytics_summary
from app.services.data_service import import_workbook, load_rows
from app.services.ticket_service import dataset_options

DATASET = PROJECT_ROOT / "data" / "raw" / "Обращения_1931.xlsx"


pytestmark = pytest.mark.usefixtures("prepared_service")


def test_import_is_idempotent() -> None:
    before = len(load_rows(DATABASE_PATH))
    result = import_workbook(DATASET, DATABASE_PATH)
    assert result["status"] == "unchanged"
    assert len(load_rows(DATABASE_PATH)) == before


def test_model_artifacts_reload_and_predict_valid_outputs() -> None:
    runtime = load_runtime(MODELS_DIR)
    response = analyze_ticket(
        runtime,
        {
            "description": "Не подключается QR-код. QR-код не отображается и невозможно получить отправление",
            "service": "",
            "component": "",
            "request_type": "",
            "criticality": "",
            "urgency": "",
            "priority": "",
            "service_class": "",
            "timezone": "",
        },
        DATABASE_PATH,
    )
    assert 0 <= response["category"]["confidence"] <= 1
    assert len(response["category"]["alternatives"]) == 3
    assert response["category"]["accepted"] is True
    assert response["category"]["label"] in runtime["metadata"]["top15"]
    assert response["routing"]["label"] in {"(1 линия)", "(2 линия)", "(3 линия)"}
    assert len(response["similar"]) >= 3
    assert all(0 <= item["similarity"] <= 1 for item in response["similar"])
    assert response["similar"] == sorted(
        response["similar"], key=lambda row: row["similarity"], reverse=True
    )


def test_lite_unknown_like_text_is_explicitly_flagged_for_review() -> None:
    runtime = load_runtime(MODELS_DIR)
    response = analyze_ticket(
        runtime,
        {
            "description": "абракадабра квантовый телепорт океан",
            "service": "",
            "component": "",
            "request_type": "",
            "criticality": "",
            "urgency": "",
            "priority": "",
            "service_class": "",
            "timezone": "",
        },
        DATABASE_PATH,
    )
    assert response["category"]["accepted"] is False
    assert response["category"]["review_required"] is True
    assert response["category"]["confidence"] < response["category"]["threshold"]
    assert response["model_provenance"]["candidate_id"] in {
        "lite-v2-tfidf-svc-c035",
        "lite-v3-qdistill-a1-fusion-s1-c035-specialist",
        "lite-v3-qdistill-customer49-c05-w4",
    }


def test_dataset_options_include_promoted_model_taxonomy() -> None:
    categories = dataset_options(DATABASE_PATH)["categories"]
    assert "Проблема с авторизацией" in categories
    assert 'Доступ в раздел "Вакансии"' in categories


def test_analytics_filters_change_totals() -> None:
    overall = analytics_summary(AnalyticsFilters(), DATABASE_PATH)
    category = overall["breakdowns"]["categories"][0]["name"]
    filtered = analytics_summary(AnalyticsFilters(category=category), DATABASE_PATH)
    assert overall["kpis"]["tickets"] == len(load_rows(DATABASE_PATH))
    assert overall["kpis"]["overdue"] >= 25
    assert 0 < filtered["kpis"]["tickets"] < overall["kpis"]["tickets"]
