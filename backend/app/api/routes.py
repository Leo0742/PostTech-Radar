from __future__ import annotations

import json
from functools import lru_cache
from typing import Annotated, Any
from urllib.parse import unquote

from fastapi import APIRouter, Header, HTTPException, Query, Request
from fastapi.responses import FileResponse

from app.core.config import DATABASE_PATH, EVALUATION_DIR, FIGURES_DIR, MODELS_DIR
from app.ml.runtime import analyze_ticket, load_runtime, route_ticket, similar_tickets
from app.schemas import TicketAnalyzeRequest
from app.schemas.incoming import (
    CategoryDecisionRequest,
    IncomingManualRequest,
    IncomingTicketUpdateRequest,
    RouteDecisionRequest,
)
from app.schemas.tickets import ConfirmedTicketUpdateRequest
from app.services.analytics_service import AnalyticsFilters, analytics_summary
from app.services.data_service import metadata_value
from app.services.incoming_service import (
    complete_ticket,
    confirm_category,
    confirm_route,
    create_batch_from_excel,
    create_manual_ticket,
    finish_batch_analysis,
    get_batch,
    get_incoming_ticket,
    list_batch_tickets,
    list_batches,
    save_analysis,
    save_analysis_error,
    save_routing_recommendation,
    set_batch_status,
    ticket_analysis_payload,
    update_incoming_ticket,
)
from app.services.process_service import process_summary
from app.services.ticket_service import (
    dataset_options,
    list_ticket_revisions,
    list_tickets,
    ticket_detail,
    update_ticket,
)

router = APIRouter(prefix="/api")


@lru_cache(maxsize=1)
def runtime() -> dict[str, Any]:
    try:
        return load_runtime(MODELS_DIR)
    except FileNotFoundError as error:
        raise HTTPException(503, "Модели не подготовлены. Выполните ./scripts/prepare.sh") from error


@router.get("/health")
def health() -> dict[str, Any]:
    ready = DATABASE_PATH.exists() and (MODELS_DIR / "category.joblib").exists()
    return {"status": "ok" if ready else "preparing", "database": DATABASE_PATH.exists(), "models": ready}


@router.get("/dataset/summary")
def dataset_summary() -> dict[str, Any]:
    summary = metadata_value("dataset_summary")
    if summary is None:
        raise HTTPException(503, "Набор данных не импортирован")
    return summary


@router.get("/dataset/options")
def options() -> dict[str, list[str]]:
    return dataset_options()


@router.post("/tickets/analyze")
def analyze(payload: TicketAnalyzeRequest) -> dict[str, Any]:
    return analyze_ticket(runtime(), payload.model_dump())


@router.post("/tickets/similar")
def similar(payload: TicketAnalyzeRequest, limit: Annotated[int, Query(ge=1, le=10)] = 6) -> list[dict[str, Any]]:
    return similar_tickets(runtime(), payload.model_dump(), limit)


@router.post("/incoming/upload")
async def incoming_upload(
    request: Request,
    x_filename: Annotated[str | None, Header()] = None,
) -> dict[str, Any]:
    filename = unquote(x_filename) if x_filename else "incoming.xlsx"
    content = await request.body()
    if not content:
        raise HTTPException(400, "Файл пуст")
    try:
        return create_batch_from_excel(content, filename)
    except ValueError as error:
        raise HTTPException(422, str(error)) from error


@router.post("/incoming/manual")
def incoming_manual(payload: IncomingManualRequest) -> dict[str, Any]:
    try:
        return create_manual_ticket(payload.model_dump())
    except ValueError as error:
        raise HTTPException(422, str(error)) from error


@router.get("/incoming/batches")
def incoming_batches() -> list[dict[str, Any]]:
    return list_batches()


@router.get("/incoming/batches/{batch_id}")
def incoming_batch(batch_id: str) -> dict[str, Any]:
    result = get_batch(batch_id)
    if result is None:
        raise HTTPException(404, "Загрузка не найдена")
    return result


@router.get("/incoming/batches/{batch_id}/tickets")
def incoming_batch_tickets(batch_id: str) -> list[dict[str, Any]]:
    if get_batch(batch_id) is None:
        raise HTTPException(404, "Загрузка не найдена")
    return list_batch_tickets(batch_id)


@router.post("/incoming/batches/{batch_id}/analyze")
def analyze_incoming_batch(batch_id: str) -> dict[str, Any]:
    batch = get_batch(batch_id)
    if batch is None:
        raise HTTPException(404, "Загрузка не найдена")
    set_batch_status(batch_id, "analyzing")
    for ticket in list_batch_tickets(batch_id):
        if ticket["state"] != "uploaded":
            continue
        try:
            payload = ticket_analysis_payload(ticket["id"])
            if payload is None:
                continue
            save_analysis(ticket["id"], analyze_ticket(runtime(), payload))
        except Exception:
            save_analysis_error(ticket["id"], "Не удалось выполнить анализ этого обращения")
    result = finish_batch_analysis(batch_id)
    if result is None:
        raise HTTPException(404, "Загрузка не найдена")
    return result


@router.get("/incoming/tickets/{ticket_id}")
def incoming_ticket(ticket_id: int) -> dict[str, Any]:
    result = get_incoming_ticket(ticket_id)
    if result is None:
        raise HTTPException(404, "Обращение не найдено")
    return result


@router.patch("/incoming/tickets/{ticket_id}")
def incoming_update_ticket(ticket_id: int, payload: IncomingTicketUpdateRequest) -> dict[str, Any]:
    try:
        result = update_incoming_ticket(ticket_id, payload.model_dump(exclude_unset=True))
    except ValueError as error:
        raise HTTPException(409, str(error)) from error
    if result is None:
        raise HTTPException(404, "Обращение не найдено")
    return result


@router.post("/incoming/tickets/{ticket_id}/analyze")
def analyze_incoming_ticket(ticket_id: int) -> dict[str, Any]:
    ticket = get_incoming_ticket(ticket_id)
    if ticket is None:
        raise HTTPException(404, "Обращение не найдено")
    if ticket["state"] == "saved":
        raise HTTPException(409, "Сохранённое обращение нельзя анализировать повторно")
    payload = ticket_analysis_payload(ticket_id)
    if payload is None:
        raise HTTPException(404, "Обращение не найдено")
    try:
        save_analysis(ticket_id, analyze_ticket(runtime(), payload))
    except HTTPException:
        raise
    except Exception as error:
        save_analysis_error(ticket_id, "Не удалось выполнить повторный анализ обращения")
        raise HTTPException(500, "Не удалось выполнить повторный анализ обращения") from error
    result = get_incoming_ticket(ticket_id)
    if result is None:
        raise HTTPException(404, "Обращение не найдено")
    return result


@router.post("/incoming/tickets/{ticket_id}/confirm-category")
def incoming_confirm_category(ticket_id: int, payload: CategoryDecisionRequest) -> dict[str, Any]:
    try:
        result = confirm_category(ticket_id, payload.category)
        if result is not None:
            analysis_payload = ticket_analysis_payload(ticket_id)
            if analysis_payload is not None:
                routing = route_ticket(
                    runtime(),
                    analysis_payload,
                    confirmed_category=payload.category,
                    database=DATABASE_PATH,
                )
                result = save_routing_recommendation(ticket_id, routing) or result
    except ValueError as error:
        raise HTTPException(409, str(error)) from error
    if result is None:
        raise HTTPException(404, "Обращение не найдено")
    return result


@router.post("/incoming/tickets/{ticket_id}/confirm-route")
def incoming_confirm_route(ticket_id: int, payload: RouteDecisionRequest) -> dict[str, Any]:
    try:
        result = confirm_route(ticket_id, payload.route)
    except ValueError as error:
        raise HTTPException(409, str(error)) from error
    if result is None:
        raise HTTPException(404, "Обращение не найдено")
    return result


@router.post("/incoming/tickets/{ticket_id}/complete")
def incoming_complete(ticket_id: int) -> dict[str, Any]:
    try:
        result = complete_ticket(ticket_id)
    except ValueError as error:
        raise HTTPException(409, str(error)) from error
    if result is None:
        raise HTTPException(404, "Обращение не найдено")
    return result


@router.get("/tickets")
def tickets(
    page: Annotated[int, Query(ge=1)] = 1,
    page_size: Annotated[int, Query(ge=1, le=100)] = 20,
    query: str | None = None,
    service: str | None = None,
    category: str | None = None,
    priority: str | None = None,
    support_line: str | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
) -> dict[str, Any]:
    return list_tickets(
        page=page,
        page_size=page_size,
        query=query,
        service=service,
        category=category,
        priority=priority,
        support_line=support_line,
        date_from=date_from,
        date_to=date_to,
    )


@router.get("/tickets/{request_id}")
def ticket(request_id: str) -> dict[str, Any]:
    result = ticket_detail(request_id)
    if result is None:
        raise HTTPException(404, "Обращение не найдено")
    return result


@router.patch("/tickets/{request_id}")
def patch_ticket(
    request_id: str,
    payload: ConfirmedTicketUpdateRequest,
    x_operator: Annotated[str | None, Header()] = None,
) -> dict[str, Any]:
    try:
        result = update_ticket(
            request_id,
            payload.model_dump(exclude_unset=True),
            changed_by=x_operator or "Оператор",
        )
    except ValueError as error:
        raise HTTPException(422, str(error)) from error
    if result is None:
        raise HTTPException(404, "Обращение не найдено")
    return result


@router.get("/tickets/{request_id}/revisions")
def ticket_revisions(request_id: str) -> list[dict[str, Any]]:
    if ticket_detail(request_id) is None:
        raise HTTPException(404, "Обращение не найдено")
    return list_ticket_revisions(request_id)


@router.get("/analytics/summary")
def analytics(
    date_from: str | None = None,
    date_to: str | None = None,
    service: str | None = None,
    category: str | None = None,
    priority: str | None = None,
    support_line: str | None = None,
) -> dict[str, Any]:
    return analytics_summary(
        AnalyticsFilters(date_from, date_to, service, category, priority, support_line)
    )


@router.get("/model/metrics")
def model_metrics() -> dict[str, Any]:
    files = {
        "category": "category_metrics.json",
        "routing": "routing_metrics.json",
        "confidence": "confidence_analysis.json",
        "retrieval": "retrieval_examples.json",
        "sla": "sla_analysis.json",
        "limitations": "limitations.json",
        "comparison": "model_comparison.json",
    }
    result = {}
    for key, filename in files.items():
        path = EVALUATION_DIR / filename
        if not path.exists():
            raise HTTPException(503, "Артефакты качества не подготовлены")
        result[key] = json.loads(path.read_text(encoding="utf-8"))
    return result


@router.get("/process/summary")
def process() -> dict[str, Any]:
    return process_summary(database=DATABASE_PATH)


@router.get("/system/status")
def system_status() -> dict[str, Any]:
    import sqlite3

    connection = sqlite3.connect(DATABASE_PATH)
    ticket_count = connection.execute("SELECT COUNT(*) FROM tickets").fetchone()[0]
    last_import = connection.execute(
        "SELECT source_hash, source_path, mode, imported_at, inserted, updated, unchanged, total "
        "FROM dataset_imports ORDER BY id DESC LIMIT 1"
    ).fetchone()
    connection.close()
    models = {}
    for name in ("category", "routing", "retrieval"):
        path = EVALUATION_DIR / f"{name}_model_metadata.json"
        models[name] = json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
    v4_registry = MODELS_DIR / "v4" / "registry.json"
    models["category_v4"] = (
        json.loads(v4_registry.read_text(encoding="utf-8"))["ACTIVE_CHAMPION"]
        if v4_registry.exists()
        else None
    )
    try:
        active = runtime()
        category_bundle = active.get("category") or {}
        routing_bundle = active.get("routing") or {}
        category_metadata = dict(category_bundle.get("metadata") or {})
        routing_metadata = dict(routing_bundle.get("metadata") or {})
        models["active_runtime"] = {
            "category": {
                **category_metadata,
                "input_contract": category_bundle.get("input_contract"),
            },
            "routing": {
                **routing_metadata,
                "input_contract": routing_bundle.get("input_contract"),
            },
            "fallback_reason": active.get("v4_fallback_reason"),
        }
    except HTTPException as error:
        models["active_runtime"] = {"unavailable": True, "reason": error.detail}
    return {
        "data": {
            "ticket_count": ticket_count,
            "dataset_hash": metadata_value("source_sha256"),
            "source_path": metadata_value("source_path"),
            "imported_at": metadata_value("imported_at"),
            "schema_version": metadata_value("schema_version"),
            "last_import": dict(zip(("source_hash", "source_path", "mode", "imported_at", "inserted", "updated", "unchanged", "total"), last_import, strict=True)) if last_import else None,
        },
        "models": models,
    }


@router.get("/model/figures/{name}")
def model_figure(name: str) -> FileResponse:
    allowed = {"category_confusion_matrix.png", "routing_confusion_matrix.png"}
    if name not in allowed:
        raise HTTPException(404, "График не найден")
    return FileResponse(FIGURES_DIR / name, media_type="image/png")
