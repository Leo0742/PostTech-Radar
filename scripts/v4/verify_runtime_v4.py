from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[2]


def _compact_analysis(result: dict[str, Any], elapsed_ms: float) -> dict[str, Any]:
    return {
        "status_code": 200,
        "elapsed_ms": round(elapsed_ms, 2),
        "category": result["category"]["label"],
        "category_confidence": result["category"]["confidence"],
        "category_decision_source": result["category"].get("decision_source"),
        "routing": result["routing"]["label"],
        "routing_confidence": result["routing"]["confidence"],
        "retrieval_rejected": result["retrieval"]["rejected"],
        "similar_count": len(result["similar"]),
        "sla_risk_level": result["sla_risk"]["level"],
        "model_provenance": result.get("model_provenance", {}),
    }


def main() -> None:
    import sys

    sys.path.insert(0, str(ROOT / "backend"))
    from app.main import app

    cases = {
        "known_qr": {
            "description": "Не получается подключить QR-код в мобильном приложении, после сканирования появляется ошибка",
            "service": "Мобильное приложение",
            "priority": "Средний",
        },
        "rare_form": {
            "description": "После оформления партии не выгружается печатный бланк, кнопка скачивания не отвечает",
            "service": "Отправка",
            "priority": "Средний",
        },
        "hard_import_confusion": {
            "description": "Не удаётся импортировать zip архив со списком формы 103 в партионный приём",
            "service": "Партионная почта",
            "priority": "Высокий",
        },
        "unknown_issue": {
            "description": "После обновления экспериментального терминала квантовой телепортации посылок появляется код ZQ-991",
            "service": "Экспериментальный сервис",
            "priority": "Средний",
        },
    }
    with TestClient(app) as client:
        health = client.get("/api/health")
        status = client.get("/api/system/status")
        model_metrics = client.get("/api/model/metrics")
        process = client.get("/api/process/summary")
        analytics = client.get("/api/analytics/summary")
        analyses: dict[str, Any] = {}
        for name, payload in cases.items():
            started = time.perf_counter()
            response = client.post("/api/tickets/analyze", json=payload)
            elapsed_ms = 1000 * (time.perf_counter() - started)
            response.raise_for_status()
            analyses[name] = _compact_analysis(response.json(), elapsed_ms)

    assert health.status_code == 200 and health.json()["status"] == "ok"
    assert status.status_code == 200
    assert model_metrics.status_code == 200
    assert process.status_code == 200
    assert analytics.status_code == 200
    assert analyses["unknown_issue"]["category"] == "UNKNOWN_NEW_ISSUE"
    assert all(item["model_provenance"].get("candidate_id") for item in analyses.values())
    verification = {
        "verified_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "transport": "FastAPI TestClient against the real application and persisted v4 runtime bundle",
        "health": health.json(),
        "system_active_champion": status.json()["models"].get("category_v4"),
        "quality_endpoint": {"status_code": model_metrics.status_code},
        "process_endpoint": {"status_code": process.status_code},
        "analytics_endpoint": {"status_code": analytics.status_code},
        "cases": analyses,
        "fallback_used": any(item["model_provenance"].get("fallback") for item in analyses.values()),
    }
    destination = ROOT / "artifacts/gpu_research_v4/final/final-evaluation.json"
    final = json.loads(destination.read_text(encoding="utf-8"))
    final["runtime_verification"] = verification
    destination.write_text(json.dumps(final, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(verification, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
