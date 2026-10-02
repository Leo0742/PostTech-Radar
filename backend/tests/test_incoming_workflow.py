from __future__ import annotations

import sqlite3
from io import BytesIO
from pathlib import Path

import pytest
from app.schemas.incoming import IncomingTicketUpdateRequest
from app.services.incoming_service import (
    complete_ticket,
    confirm_category,
    confirm_route,
    create_batch_from_excel,
    get_batch,
    get_incoming_ticket,
    list_batch_tickets,
    save_analysis,
    ticket_analysis_payload,
    update_incoming_ticket,
)
from openpyxl import Workbook
from pydantic import ValidationError


def workbook_bytes(headers: list[str], rows: list[list[object]]) -> bytes:
    workbook = Workbook()
    sheet = workbook.active
    sheet.append(headers)
    for row in rows:
        sheet.append(row)
    stream = BytesIO()
    workbook.save(stream)
    return stream.getvalue()


def analysis() -> dict[str, object]:
    return {
        "category": {
            "label": "Категория модели",
            "confidence": 0.72,
            "review_required": True,
            "alternatives": [
                {"label": "Категория модели", "confidence": 0.72},
                {"label": "Другая категория", "confidence": 0.18},
            ],
        },
        "routing": {
            "label": "2 линия",
            "confidence": 0.81,
            "review_required": False,
            "alternatives": [],
        },
        "sla_risk": {"risk": 0.1, "sample_size": 8, "level": "Низкий", "explanation": ""},
        "similar": [],
    }


def test_excel_upload_accepts_partial_schema_and_marks_short_text_for_attention(tmp_path) -> None:
    database = tmp_path / "incoming.sqlite3"
    content = workbook_bytes(
        ["Номер запроса", "Описание", "Услуга"],
        [["A-1", "Не отображается QR-код для получения", "Почта"], ["A-2", "коротко", "Почта"]],
    )
    batch = create_batch_from_excel(content, "incoming.xlsx", database)
    tickets = list_batch_tickets(batch["id"], database)

    assert batch["total_rows"] == 2
    assert batch["valid_rows"] == 2
    assert batch["invalid_rows"] == 0
    assert batch["description_column"] == "Описание"
    assert "component" in batch["missing_optional"]
    assert tickets[0]["state"] == "uploaded"
    assert tickets[1]["state"] == "uploaded"
    assert tickets[1]["low_information"] is True
    assert tickets[1]["attention"] is True


def test_excel_upload_reports_unusable_rows_per_row_and_rejects_bad_extension(tmp_path) -> None:
    database = tmp_path / "incoming.sqlite3"
    batch = create_batch_from_excel(workbook_bytes(["Номер запроса"], [["A-1"]]), "incoming.xlsx", database)
    tickets = list_batch_tickets(batch["id"], database)
    assert batch["total_rows"] == 1
    assert batch["valid_rows"] == 0
    assert batch["invalid_rows"] == 1
    assert tickets[0]["state"] == "failed"
    assert "Недостаточно данных" in tickets[0]["error_text"]
    with pytest.raises(ValueError, match=".xlsx"):
        create_batch_from_excel(b"not excel", "incoming.csv", database)


def test_excel_upload_accepts_exact_official_description_2_header(tmp_path) -> None:
    database = tmp_path / "incoming.sqlite3"
    batch = create_batch_from_excel(
        workbook_bytes(
            ["Номер запроса", "  ОПИСАНИЕ   2  ", "Часовой пояс запроса"],
            [["A-1", "Проблема с получением отправления", "Москва"]],
        ),
        "official-like.xlsx",
        database,
    )
    ticket = list_batch_tickets(batch["id"], database)[0]

    assert batch["description_column"] == "ОПИСАНИЕ   2"
    assert ticket["description"] == "Проблема с получением отправления"
    assert ticket["timezone"] == "Москва"


def test_actual_organizer_1931_excel_uploads_without_schema_error(tmp_path) -> None:
    source = Path(__file__).resolve().parents[2] / "data" / "raw" / "Обращения_1931.xlsx"
    database = tmp_path / "official.sqlite3"
    batch = create_batch_from_excel(source.read_bytes(), source.name, database)

    assert batch["total_rows"] == 1931
    assert batch["valid_rows"] == 1931
    assert batch["invalid_rows"] == 0
    assert batch["description_column"] == "Описание 2"
    assert batch["labeled_historical"] is True
    assert {
        "Номер запроса",
        "Дата регистрации",
        "Пользователь",
        "Услуга",
        "Компонент услуги 1 уровня",
        "Тип запроса",
        "Описание 2",
        "Критичность",
        "Срочность",
        "Приоритет",
        "Класс обслуживания",
        "Часовой пояс запроса",
    }.issubset(set(batch["recognized_columns"]))


def test_historical_targets_and_unknown_columns_never_enter_inference_payload(tmp_path) -> None:
    database = tmp_path / "incoming.sqlite3"
    batch = create_batch_from_excel(
        workbook_bytes(
            [
                "Описание 2",
                "Услуга",
                "Вид запроса",
                "Кем решен (группа)",
                "Статус",
                "Результат работ",
                "Неизвестная колонка заказчика",
            ],
            [["Не работает приложение", "Мобильное приложение", "Готовая категория", "3 линия", "Закрыт", "Готово", "extra"]],
        ),
        "historical.xlsx",
        database,
    )
    ticket = list_batch_tickets(batch["id"], database)[0]
    payload = ticket_analysis_payload(ticket["id"], database)

    assert batch["labeled_historical"] is True
    assert payload == {
        "description": "Не работает приложение",
        "service": "Мобильное приложение",
        "component": "",
        "request_type": "",
        "criticality": "",
        "urgency": "",
        "priority": "",
        "service_class": "",
        "timezone": "",
        "registration_date": "",
        "user": "",
    }
    assert ticket["raw"]["Вид запроса"] == "Готовая категория"
    assert "Вид запроса" not in payload
    assert "Кем решен (группа)" not in payload
    assert "Статус" not in payload
    assert "Результат работ" not in payload


def test_metadata_only_row_is_accepted_and_marked_low_information(tmp_path) -> None:
    database = tmp_path / "incoming.sqlite3"
    batch = create_batch_from_excel(
        workbook_bytes(["Номер запроса", "Услуга", "Приоритет"], [["M-1", "Почта", "Высокий"]]),
        "metadata-only.xlsx",
        database,
    )
    ticket = list_batch_tickets(batch["id"], database)[0]

    assert batch["description_column"] is None
    assert batch["valid_rows"] == 1
    assert ticket["state"] == "uploaded"
    assert ticket["description"] == ""
    assert ticket["low_information"] is True
    assert ticket_analysis_payload(ticket["id"], database)["service"] == "Почта"


def test_operator_review_is_persisted_and_does_not_auto_retrain(tmp_path) -> None:
    database = tmp_path / "incoming.sqlite3"
    batch = create_batch_from_excel(
        workbook_bytes(["Номер запроса", "Описание", "Приоритет"], [["NEW-101", "Не работает получение отправления по QR-коду", "Высокий"]]),
        "daily.xlsx",
        database,
    )
    ticket = list_batch_tickets(batch["id"], database)[0]
    save_analysis(ticket["id"], analysis(), database)

    first = confirm_category(ticket["id"], "Исправленная категория", database)
    assert first and first["category_confirmed"] == 1
    routed = confirm_route(ticket["id"], "3 линия", database)
    assert routed and routed["route_confirmed"] == 1

    changed_again = confirm_category(ticket["id"], "Итоговая категория", database)
    assert changed_again and changed_again["route_confirmed"] == 0
    confirm_route(ticket["id"], "3 линия", database)
    saved = complete_ticket(ticket["id"], database)
    assert saved == {"status": "saved", "request_id": "NEW-101", "incoming_ticket_id": ticket["id"]}
    assert complete_ticket(ticket["id"], database) == saved

    connection = sqlite3.connect(database)
    row = connection.execute("SELECT category, final_line, source_hash FROM tickets WHERE request_id='NEW-101'").fetchone()
    review = connection.execute("SELECT operator_final_category, operator_final_route, category_changed, route_changed FROM ticket_reviews").fetchone()
    metadata = connection.execute("SELECT value FROM metadata WHERE key='dataset_summary'").fetchone()
    connection.close()

    assert row == ("Итоговая категория", "3 линия", "operator-review")
    assert review == ("Итоговая категория", "3 линия", 1, 1)
    assert metadata is None
    assert get_incoming_ticket(ticket["id"], database)["state"] == "saved"
    assert get_batch(batch["id"], database)["status"] == "saved"


def test_source_edit_updates_only_registration_fields_and_invalidates_decisions(tmp_path) -> None:
    database = tmp_path / "incoming.sqlite3"
    batch = create_batch_from_excel(
        workbook_bytes(
            ["Номер запроса", "Описание", "Услуга", "Пользователь"],
            [["EDIT-1", "Исходное описание обращения", "Почта", "Иван"]],
        ),
        "daily.xlsx",
        database,
    )
    ticket = list_batch_tickets(batch["id"], database)[0]
    original_raw = ticket["raw"]
    save_analysis(ticket["id"], analysis(), database)
    confirm_category(ticket["id"], "Категория модели", database)
    confirm_route(ticket["id"], "2 линия", database)

    edited = update_incoming_ticket(
        ticket["id"],
        {"description": "Исправленное описание обращения", "service": "Новая услуга"},
        database,
    )
    assert edited is not None
    assert edited["description"] == "Исправленное описание обращения"
    assert edited["service"] == "Новая услуга"
    assert edited["user_name"] == "Иван"
    assert edited["state"] == "uploaded"
    assert edited["analysis"] is None
    assert edited["model_category"] is None
    assert edited["model_route"] is None
    assert edited["operator_category"] is None
    assert edited["operator_route"] is None
    assert edited["category_confirmed"] == 0
    assert edited["route_confirmed"] == 0
    assert edited["raw"] == original_raw

    connection = sqlite3.connect(database)
    events = connection.execute(
        "SELECT event_type FROM incoming_ticket_audit WHERE incoming_ticket_id=? ORDER BY id",
        (ticket["id"],),
    ).fetchall()
    connection.close()
    assert [item[0] for item in events] == [
        "analysis",
        "category_confirmed",
        "route_confirmed",
        "source_edit",
    ]


def test_reanalysis_payload_uses_current_edited_values_and_audit_keeps_both_predictions(tmp_path) -> None:
    database = tmp_path / "incoming.sqlite3"
    batch = create_batch_from_excel(
        workbook_bytes(["Номер запроса", "Описание", "Приоритет"], [["EDIT-2", "Старый текст обращения", "Низкий"]]),
        "daily.xlsx",
        database,
    )
    ticket = list_batch_tickets(batch["id"], database)[0]
    save_analysis(ticket["id"], analysis(), database)
    update_incoming_ticket(
        ticket["id"],
        {"description": "Новый текст обращения для повторного анализа", "priority": "Высокий"},
        database,
    )
    payload = ticket_analysis_payload(ticket["id"], database)
    assert payload is not None
    assert payload["description"] == "Новый текст обращения для повторного анализа"
    assert payload["priority"] == "Высокий"

    second = analysis()
    second["category"]["label"] = "Новая категория"
    second["category"]["confidence"] = 0.91
    save_analysis(ticket["id"], second, database)

    connection = sqlite3.connect(database)
    audit_rows = connection.execute(
        "SELECT event_type, snapshot_json FROM incoming_ticket_audit WHERE incoming_ticket_id=? ORDER BY id",
        (ticket["id"],),
    ).fetchall()
    connection.close()
    assert [row[0] for row in audit_rows] == ["analysis", "source_edit", "analysis"]
    assert '"model_category": "Категория модели"' in audit_rows[1][1]
    assert '"model_category": "Новая категория"' in audit_rows[2][1]


def test_duplicate_request_id_is_rejected_and_saved_ticket_cannot_be_edited(tmp_path) -> None:
    database = tmp_path / "incoming.sqlite3"
    batch = create_batch_from_excel(
        workbook_bytes(
            ["Номер запроса", "Описание"],
            [["DUP-1", "Первое обращение с достаточным текстом"], ["DUP-2", "Второе обращение с достаточным текстом"]],
        ),
        "daily.xlsx",
        database,
    )
    first, second = list_batch_tickets(batch["id"], database)
    with pytest.raises(ValueError, match="уже используется"):
        update_incoming_ticket(second["id"], {"request_id": "DUP-1"}, database)

    save_analysis(first["id"], analysis(), database)
    confirm_category(first["id"], "Категория модели", database)
    confirm_route(first["id"], "2 линия", database)
    complete_ticket(first["id"], database)
    with pytest.raises(ValueError, match="нельзя изменять"):
        update_incoming_ticket(first["id"], {"description": "Попытка изменения"}, database)


def test_patch_schema_forbids_model_and_post_resolution_fields() -> None:
    with pytest.raises(ValidationError):
        IncomingTicketUpdateRequest(description="Текст", model_category="Нельзя")  # type: ignore[call-arg]
    with pytest.raises(ValidationError):
        IncomingTicketUpdateRequest(description="Текст", final_line="4 линия")  # type: ignore[call-arg]
