from __future__ import annotations

from app.api import routes
from app.main import app
from fastapi.testclient import TestClient

client = TestClient(app)


def test_health_and_dataset_summary() -> None:
    health = client.get("/api/health")
    assert health.status_code == 200
    assert health.json()["status"] == "ok"
    summary = client.get("/api/dataset/summary")
    assert summary.status_code == 200
    assert summary.json()["records"] >= 1931


def test_analyze_and_similar_flow() -> None:
    payload = {
        "description": "Не подключается QR-код. QR-код не отображается и невозможно получить отправление",
        "service": "",
        "component": "",
        "request_type": "",
        "criticality": "",
        "urgency": "",
        "priority": "",
        "service_class": "",
        "timezone": "",
    }
    response = client.post("/api/tickets/analyze", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body["category"]["label"] == "Проблема с QR-код(подключение/отключение)"
    assert body["similar"]
    similar = client.post("/api/tickets/similar?limit=5", json=payload)
    assert similar.status_code == 200
    assert len(similar.json()) == 5


def test_ticket_list_and_original_detail() -> None:
    listing = client.get("/api/tickets?page=1&page_size=10&query=QR")
    assert listing.status_code == 200
    body = listing.json()
    assert body["items"]
    assert "request_type" in body["items"][0]
    request_id = body["items"][0]["request_id"]
    detail = client.get(f"/api/tickets/{request_id}")
    assert detail.status_code == 200
    assert detail.json()["request_id"] == request_id
    assert "raw" in detail.json()
    assert client.get("/api/tickets/999999999999").status_code == 404


def test_invalid_filter_dates_return_validation_error() -> None:
    assert client.get("/api/tickets", params={"date_from": "not-a-date"}).status_code == 422
    assert client.get("/api/analytics/summary", params={"date_to": "2026-99-99"}).status_code == 422


def test_v2_process_and_system_status_endpoints() -> None:
    process = client.get("/api/process/summary")
    assert process.status_code == 200
    assert process.json()["status_history"]["available"] is False
    assert process.json()["participation"]["edge_semantics"].startswith("участие")
    status = client.get("/api/system/status")
    assert status.status_code == 200
    summary = client.get("/api/dataset/summary")
    assert summary.status_code == 200
    assert status.json()["data"]["ticket_count"] == summary.json()["records"]
    assert status.json()["models"]["category"]["training_record_count"] == summary.json()["top15_records"]


def test_analytics_filters_and_model_metrics() -> None:
    base = client.get("/api/analytics/summary").json()
    category = base["breakdowns"]["categories"][0]["name"]
    filtered = client.get("/api/analytics/summary", params={"category": category}).json()
    assert filtered["kpis"]["tickets"] < base["kpis"]["tickets"]
    metrics = client.get("/api/model/metrics")
    assert metrics.status_code == 200
    assert metrics.json()["category"]["macro_f1"] > 0


def test_invalid_analysis_request_is_rejected() -> None:
    response = client.post("/api/tickets/analyze", json={"request_id": "REQ-ONLY"})
    assert response.status_code == 422


def test_qwen_recheck_endpoint_is_explicit_and_separate(monkeypatch) -> None:
    monkeypatch.setattr(
        routes,
        "recheck_with_qwen",
        lambda payload: {
            "category": {
                "label": "Личный кабинет",
                "confidence": 0.71,
                "alternatives": [
                    {"label": "Личный кабинет", "confidence": 0.71},
                    {"label": "Недоступность портала", "confidence": 0.2},
                ],
            },
            "model_provenance": {"model_id": "Qwen/Qwen3-Embedding-4B", "profile": "qwen_recheck"},
            "latency_seconds": 42.0,
        },
    )

    response = client.post(
        "/api/tickets/recheck-qwen",
        json={"description": "не могу войти в личный кабинет"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["category"]["label"] == "Личный кабинет"
    assert body["model_provenance"]["profile"] == "qwen_recheck"
