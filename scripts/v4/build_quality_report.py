from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)

ROOT = Path(__file__).resolve().parents[2]
ART = ROOT / "artifacts/gpu_research_v4"
OUTPUT = ROOT / "outputs/PostTech_Radar_v4_quality_report.pdf"

NAVY = colors.HexColor("#071A3A")
BLUE = colors.HexColor("#1467E8")
CYAN = colors.HexColor("#31B7FF")
ORANGE = colors.HexColor("#FF8A34")
GREEN = colors.HexColor("#19B37A")
INK = colors.HexColor("#0B1529")
MUTED = colors.HexColor("#526174")
GRID = colors.HexColor("#D7E3F4")
PALE = colors.HexColor("#EDF4FF")


def load(path: Path, default: Any | None = None) -> Any:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else (default or {})


def at(value: Any, *keys: str, default: Any = None) -> Any:
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return default
        value = value[key]
    return value


def pct(value: Any, digits: int = 2) -> str:
    return "—" if value is None else f"{100 * float(value):.{digits}f}%"


def num(value: Any, digits: int = 3) -> str:
    return "—" if value is None else f"{float(value):.{digits}f}"


def safe(value: Any) -> str:
    return str(value if value is not None else "—").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _font_path(*candidates: str) -> str:
    for candidate in candidates:
        if Path(candidate).exists():
            return candidate
    raise FileNotFoundError(f"Cyrillic font not found: {candidates}")


pdfmetrics.registerFont(
    TTFont(
        "Verdana",
        _font_path(
            "/System/Library/Fonts/Supplemental/Verdana.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        ),
    )
)
pdfmetrics.registerFont(
    TTFont(
        "Verdana-Bold",
        _font_path(
            "/System/Library/Fonts/Supplemental/Verdana Bold.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        ),
    )
)

styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name="CoverKicker", fontName="Verdana-Bold", fontSize=10, leading=14, textColor=CYAN, spaceAfter=16))
styles.add(ParagraphStyle(name="CoverTitle", fontName="Verdana-Bold", fontSize=33, leading=38, textColor=colors.white, spaceAfter=18))
styles.add(ParagraphStyle(name="CoverSub", fontName="Verdana", fontSize=14, leading=21, textColor=colors.HexColor("#CFE1F7")))
styles.add(ParagraphStyle(name="H1R", fontName="Verdana-Bold", fontSize=21, leading=27, textColor=NAVY, spaceBefore=10, spaceAfter=12))
styles.add(ParagraphStyle(name="H2R", fontName="Verdana-Bold", fontSize=14, leading=20, textColor=BLUE, spaceBefore=10, spaceAfter=7))
styles.add(ParagraphStyle(name="BodyR", fontName="Verdana", fontSize=9.3, leading=14.2, textColor=INK, spaceAfter=7))
styles.add(ParagraphStyle(name="SmallR", fontName="Verdana", fontSize=7.5, leading=11, textColor=MUTED))
styles.add(ParagraphStyle(name="CalloutR", fontName="Verdana-Bold", fontSize=11, leading=17, textColor=NAVY, backColor=PALE, borderColor=GRID, borderWidth=0.7, borderPadding=9, spaceBefore=8, spaceAfter=10))
styles.add(ParagraphStyle(name="TableHeadR", fontName="Verdana-Bold", fontSize=7.2, leading=9, textColor=colors.white, alignment=TA_CENTER))
styles.add(ParagraphStyle(name="TableCellR", fontName="Verdana", fontSize=6.8, leading=9, textColor=INK, alignment=TA_LEFT))
styles.add(ParagraphStyle(name="MetricR", fontName="Verdana-Bold", fontSize=16, leading=20, textColor=BLUE, alignment=TA_CENTER))
styles.add(ParagraphStyle(name="MetricLabelR", fontName="Verdana", fontSize=7.5, leading=10, textColor=MUTED, alignment=TA_CENTER))


def header_footer(canvas: Any, doc: Any) -> None:
    canvas.saveState()
    if doc.page > 1:
        canvas.setStrokeColor(GRID)
        canvas.line(18 * mm, 282 * mm, 192 * mm, 282 * mm)
        canvas.setFont("Verdana-Bold", 7.5)
        canvas.setFillColor(BLUE)
        canvas.drawString(18 * mm, 286 * mm, "ПОЧТАТЕХ РАДАР · ОТЧЁТ О КАЧЕСТВЕ V4")
        canvas.setFont("Verdana", 7.5)
        canvas.setFillColor(MUTED)
        canvas.drawRightString(192 * mm, 10 * mm, f"{doc.page}")
    canvas.restoreState()


def metric_table(items: list[tuple[str, str]]) -> Table:
    cells = [
        [Paragraph(value, styles["MetricR"]), Paragraph(label, styles["MetricLabelR"])]
        for value, label in items
    ]
    table = Table([cells], colWidths=[174 * mm / len(items)] * len(items))
    table.setStyle(
        TableStyle(
            [
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("BOX", (0, 0), (-1, -1), 0.6, GRID),
                ("INNERGRID", (0, 0), (-1, -1), 0.6, GRID),
                ("BACKGROUND", (0, 0), (-1, -1), colors.white),
                ("TOPPADDING", (0, 0), (-1, -1), 8),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
            ]
        )
    )
    return table


def data_table(headers: list[str], rows: list[list[Any]], widths: list[float] | None = None) -> Table:
    values = [[Paragraph(safe(value), styles["TableHeadR"]) for value in headers]]
    for row in rows:
        values.append([Paragraph(safe(value), styles["TableCellR"]) for value in row])
    table = Table(values, colWidths=widths, repeatRows=1, hAlign="LEFT")
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), NAVY),
                ("BOX", (0, 0), (-1, -1), 0.5, GRID),
                ("INNERGRID", (0, 0), (-1, -1), 0.35, GRID),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F7FAFE")]),
            ]
        )
    )
    return table


def section(story: list[Any], title: str, intro: str | None = None) -> None:
    story.append(Paragraph(title, styles["H1R"]))
    if intro:
        story.append(Paragraph(intro, styles["BodyR"]))


def paragraph(story: list[Any], text: str, style: str = "BodyR") -> None:
    story.append(Paragraph(text, styles[style]))


def build() -> None:
    final = load(ART / "final/final-evaluation.json")
    if final.get("status") != "FINAL_SEALED_EVALUATION_COMPLETE":
        raise RuntimeError("quality report requires the completed sealed evaluation")
    selection = load(ART / "final/selection.json")
    environment = load(ART / "server_environment.json")
    pointer = load(ART / "protocol.json")
    protocol = load(ROOT / str(pointer.get("protocol_path", "")))
    synthetic = load(ART / "deep/synthetic-recheck-full43.json")
    synthetic_finalists = load(ART / "deep/synthetic-finalists.json")
    synthetic_manifest = load(ART / "synthetic/domain_corpus_clean_manifest.json")
    audit = load(ART / "synthetic/manual_synthetic_audit.json")
    ood = load(ART / "ood/leave-category-out.json")
    routing = load(ART / "routing/oof-category.json")
    retrieval = load(ART / "retrieval/qwen3-06b-qwen-reranker-deep.json")
    neural = load(ART / "deep/neural-finalist-summary.json")
    coverage = load(ART / "final/candidate-coverage-audit.json")

    top = at(final, "metrics", "top15", "metrics", default={})
    full = at(final, "metrics", "full43", "metrics", default={})
    champion = at(final, "registry", "ACTIVE_CHAMPION", default={})
    verification = final.get("runtime_verification", {})

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    document = BaseDocTemplate(
        str(OUTPUT),
        pagesize=A4,
        leftMargin=18 * mm,
        rightMargin=18 * mm,
        topMargin=20 * mm,
        bottomMargin=16 * mm,
        title="ПочтаТех Радар — отчёт о качестве решения v4",
        author="Команда ПочтаТех Радар",
        subject="Мониторинг и категоризация обращений Service Desk",
    )
    frame = Frame(document.leftMargin, document.bottomMargin, document.width, document.height, id="main")
    document.addPageTemplates([PageTemplate(id="report", frames=[frame], onPage=header_footer)])

    story: list[Any] = []
    # Cover uses a full-width dark table to remain robust across PDF renderers.
    cover = Table(
        [[
            Paragraph(
                "POSTCODE CHALLENGE · ТРЕК 3<br/><br/><font size='33'><b>ПочтаТех Радар</b></font><br/><br/>"
                "Отчёт о качестве решения<br/><br/><font size='10'>Финальная версия v4 · real-only sealed evaluation</font>",
                styles["CoverSub"],
            )
        ]],
        colWidths=[174 * mm],
        rowHeights=[235 * mm],
    )
    cover.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), NAVY),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 18 * mm),
                ("RIGHTPADDING", (0, 0), (-1, -1), 18 * mm),
                ("BOX", (0, 0), (-1, -1), 0, NAVY),
            ]
        )
    )
    story.extend([Spacer(1, 5 * mm), cover, PageBreak()])

    section(story, "1. Резюме результата")
    paragraph(
        story,
        "Сервис локально классифицирует обращения, рекомендует линию поддержки, находит похожие решённые кейсы и показывает аналитику SLA. Финальный выбор сделан до единственного доступа к sealed holdout. Валидация, калибровка, temporal и финальный тест содержат только реальные тикеты.",
    )
    story.append(
        metric_table(
            [
                (pct(top.get("accuracy")), "TOP15 accuracy"),
                (pct(top.get("macro_f1")), "TOP15 macro-F1"),
                (pct(full.get("accuracy")), "FULL43 accuracy"),
                (pct(full.get("macro_f1")), "FULL43 macro-F1"),
            ]
        )
    )
    story.append(Spacer(1, 7 * mm))
    paragraph(story, f"Production champion: <b>{safe(champion.get('candidate_id', final.get('candidate_id')))}</b>.", "CalloutR")
    paragraph(
        story,
        f"FULL43 balanced accuracy: <b>{pct(full.get('balanced_accuracy'))}</b>. Worst-class F1: <b>{pct(full.get('worst_class_f1'))}</b>. Измеренная задержка: <b>{num(at(final, 'metrics', 'full43', 'latency_ms_per_ticket_including_encoder'), 1)} мс/обращение</b>.",
    )

    section(story, "2. Данные и воспроизводимый протокол")
    protocol_rows = [
        ["Реальные строки", protocol.get("real_row_count")],
        ["Категории", len(protocol.get("labels", []))],
        ["Frozen grouped-CV folds", len(protocol.get("folds", []))],
        ["Dataset SHA-256", protocol.get("dataset_sha256")],
        ["Split SHA-256", protocol.get("split_sha256")],
        ["Sealed holdout", "прочитан один раз после фиксации selection.json"],
    ]
    story.append(data_table(["Параметр", "Значение"], protocol_rows, [48 * mm, 126 * mm]))
    paragraph(story, "Нормализованные дубликаты удерживаются в одной группе. Все признаки соответствуют полям, доступным при регистрации. Итоговые статусы, решение и время работы не участвуют в классификации нового обращения.")

    section(story, "3. GPU-среда и фактически измеренные семейства")
    env_rows = [
        ["GPU", environment.get("gpu"), f"{environment.get('vram_gib', '—')} GiB"],
        ["CPU / RAM", environment.get("cpu"), f"{environment.get('ram_gib', '—')} GiB"],
        ["ПО", environment.get("os"), f"PyTorch {environment.get('pytorch', '—')}"],
        ["CUDA", environment.get("cuda_driver_runtime"), f"driver {environment.get('nvidia_driver', '—')}"],
    ]
    story.append(data_table(["Компонент", "Среда", "Версия/объём"], env_rows, [34 * mm, 92 * mm, 48 * mm]))
    paragraph(
        story,
        f"Candidate coverage до finalist selection: <b>{safe(coverage.get('measured_family_count'))}/"
        f"{safe(coverage.get('planned_family_count'))}</b> семейств; sealed holdout на этом этапе: "
        f"<b>{'закрыт' if coverage.get('sealed_holdout_accessed') is False else 'статус не подтверждён'}</b>.",
        "CalloutR",
    )
    candidate_rows = []
    for item in selection.get("candidates", []):
        candidate_rows.append(
            [
                item.get("family"),
                pct(at(item, "top15", "macro_f1")),
                pct(at(item, "full43", "macro_f1")),
                pct(item.get("full43_rare_1_5_macro_f1_mean")),
                item.get("latency_ms_screening"),
                item.get("vram_gib_screening"),
                num(item.get("production_score"), 4),
            ]
        )
    story.append(Spacer(1, 4 * mm))
    story.append(data_table(["Семейство", "TOP15 F1", "FULL43 F1", "Rare F1", "мс", "VRAM", "Prod score"], candidate_rows, [40 * mm, 22 * mm, 24 * mm, 22 * mm, 17 * mm, 20 * mm, 25 * mm]))
    paragraph(story, "Дополнительно реально запускались ruBert-base, research-only ruRoberta-large, DeepPavlov AutoIntent, FastFit/MPNet, FastFit/XLM-R base, XLM-R large LoRA и full fine-tuning, label-semantic mDeBERTa NLI, BGE-M3, multilingual E5, Qwen3 embeddings и OOF stacking. Отрицательные результаты сохранены в артефактах и не скрыты.")

    story.append(PageBreak())
    section(story, "4. TOP15 и FULL43")
    story.append(
        metric_table(
            [
                (pct(top.get("accuracy")), "TOP15 accuracy"),
                (pct(top.get("macro_f1")), "TOP15 macro-F1"),
                (pct(full.get("balanced_accuracy")), "FULL43 balanced accuracy"),
                (pct(full.get("worst_class_f1")), "FULL43 worst-class F1"),
            ]
        )
    )
    paragraph(story, "TOP15 соответствует бизнес-требованию основной категоризации. FULL43 показывает способность архитектуры работать со всей таксономией и не маскирует редкие классы средним accuracy.")
    support_rows = []
    for band, values in (full.get("support_bands") or {}).items():
        support_rows.append([band, values.get("class_count"), pct(values.get("macro_f1"))])
    if support_rows:
        story.append(data_table(["Support band", "Классов", "Macro-F1"], support_rows, [58 * mm, 46 * mm, 50 * mm]))

    section(story, "5. Метрики по классам FULL43")
    per_class_rows = []
    for label, values in (full.get("per_class") or {}).items():
        per_class_rows.append([label, values.get("support"), pct(values.get("precision")), pct(values.get("recall")), pct(values.get("f1"))])
    if per_class_rows:
        story.append(data_table(["Категория", "Support", "Precision", "Recall", "F1"], per_class_rows, [86 * mm, 20 * mm, 23 * mm, 21 * mm, 20 * mm]))

    story.append(PageBreak())
    section(story, "6. Редкие классы и synthetic TRAIN")
    baseline = at(synthetic, "baseline", "metrics", default={})
    best = at(synthetic, "best", "metrics", default={})
    story.append(
        metric_table(
            [
                (str(synthetic_manifest.get("rows_after_qc", "—")), "строк после QC"),
                (pct(baseline.get("macro_f1")), "real baseline macro-F1"),
                (pct(best.get("macro_f1")), "лучший synthetic macro-F1"),
                (pct(at(best, "support_bands", "1-5", "macro_f1")), "support 1–5 F1"),
            ]
        )
    )
    paragraph(story, f"Ручной semantic audit: <b>{safe(audit.get('status'))}</b>. Одобрено категорий: <b>{len(audit.get('approved_categories', []))}</b>; отклонено категорий: <b>{len(audit.get('rejected_categories', []))}</b>.")
    paragraph(story, "ZERO-real классы получают метку SYNTHETIC_ONLY и эффективный вес не выше 0,05 при любом scale. ONE-real классы используют единственный реальный якорь, генерируют разнообразие вместо парафразов и имеют вес не выше 0,10. Их метрики публикуются отдельно.", "CalloutR")
    family_rows = []
    for family, values in synthetic_finalists.get("families", {}).items():
        family_rows.append([family, values.get("accepted"), pct(at(values, "baseline", "metrics", "macro_f1")), pct(at(values, "best", "metrics", "macro_f1")), at(values, "best", "ratio"), at(values, "best", "weight_scale")])
    if family_rows:
        story.append(data_table(["Семейство", "Принят", "Baseline F1", "Best F1", "Ratio", "Scale"], family_rows, [44 * mm, 19 * mm, 30 * mm, 27 * mm, 24 * mm, 24 * mm]))

    section(story, "7. Нейросетевые финалисты")
    neural_rows = []
    for candidate, values in neural.get("candidates", {}).items():
        neural_rows.append([candidate, values.get("run_count"), pct(at(values, "stability", "accuracy", "mean")), pct(at(values, "stability", "macro_f1", "mean")), pct(at(values, "stability", "macro_f1", "std"))])
    if neural_rows:
        story.append(data_table(["Кандидат", "Runs", "Accuracy", "Macro-F1", "F1 std"], neural_rows, [83 * mm, 18 * mm, 25 * mm, 25 * mm, 23 * mm]))
    paragraph(story, "Нейросетевые результаты оцениваются как deep-finalist evidence, но не получают автоматический production статус без полного 12-fold/runtime контракта. Это исключает несправедливое сравнение одного shallow запуска с глубоко оптимизированной классической моделью.")

    story.append(PageBreak())
    section(story, "8. Маршрутизация, retrieval и OOD")
    adjacent_rows = [
        ["Маршрутизация", pct(at(routing, "metrics", "accuracy", default=at(routing, "accuracy"))), pct(at(routing, "metrics", "macro_f1", default=at(routing, "macro_f1"))), "OOF category feature"],
        ["Retrieval", num(at(retrieval, "sealed", "mrr", default=at(retrieval, "metrics", "mrr"))), num(at(retrieval, "sealed", "recall_at_5", default=at(retrieval, "metrics", "recall_at_5"))), "sealed retrieval split"],
        ["OOD/UNKNOWN", pct(at(ood, "metrics", "ood_rejection", default=at(ood, "ood_rejection"))), pct(at(ood, "metrics", "known_acceptance", default=at(ood, "known_acceptance"))), "leave-category-out"],
    ]
    story.append(data_table(["Задача", "Метрика 1", "Метрика 2", "Протокол"], adjacent_rows, [43 * mm, 32 * mm, 32 * mm, 67 * mm]))
    paragraph(story, "Роутинг оценивается отдельно от категории. Retrieval исключает self-hit и точные нормализованные дубликаты. UNKNOWN_NEW_ISSUE является финальным API-результатом, когда calibrated confidence ниже порога.")

    section(story, "9. Runtime verification")
    cases = verification.get("cases", {})
    case_rows = []
    for name, values in cases.items():
        case_rows.append([name, values.get("category"), values.get("routing"), values.get("similar_count"), values.get("sla_risk_level"), values.get("elapsed_ms")])
    if case_rows:
        story.append(data_table(["Кейс", "Категория", "Маршрут", "Similar", "SLA", "мс"], case_rows, [30 * mm, 55 * mm, 32 * mm, 16 * mm, 18 * mm, 18 * mm]))
    paragraph(story, f"Health: <b>{safe(at(verification, 'health', 'status'))}</b>. Fallback used: <b>{safe(verification.get('fallback_used'))}</b>. Проверка выполнила реальный FastAPI pipeline и persisted v4 bundle.")

    section(story, "10. Архитектура и масштабирование")
    paragraph(story, "FastAPI обслуживает классификацию, маршрутизацию, retrieval, SLA и provenance. SQLite хранит историю и поддерживает cumulative upsert. React/TypeScript показывает операторский сценарий, аналитику, процесс, качество и происхождение моделей. Внешние AI API в runtime отсутствуют.")
    paragraph(story, "При росте до 10k/100k+ реальные подтверждённые исходы формируют новую версию датасета. Новый hash создаёт новый frozen protocol. Challenger сравнивается с ACTIVE_CHAMPION на grouped и temporal тестах и активируется только после quality/latency gate.")

    section(story, "11. Ограничения")
    limitations = [
        "Классы с 0–1 реальным TRAIN примером не имеют устойчивого supervised доказательства; синтетика не меняет этот факт.",
        "Для retrieval нет полной независимой ручной релевантностной разметки; CSV для экспертного review сохранён отдельно.",
        "GigaChat/Yandex API не тестировались: учётные данные не предоставлены, а локальный контур не зависит от них.",
        "Внешнее GitLab-размещение не выполнялось и требует отдельной авторизации пользователя.",
        "После передачи артефактов скачанные model caches удаляются с арендованного GPU-сервера по запросу владельца.",
    ]
    for item in limitations:
        paragraph(story, f"• {item}")

    section(story, "12. Состав доказательств")
    evidence_rows = [
        ["Выбор победителя", "artifacts/gpu_research_v4/final/selection.json"],
        ["Sealed evaluation", "artifacts/gpu_research_v4/final/final-evaluation.json"],
        ["Synthetic audit", "artifacts/gpu_research_v4/synthetic/manual_synthetic_audit.json"],
        ["Candidate coverage", "artifacts/gpu_research_v4/final/candidate-coverage-audit.json"],
        ["Model registry", "models/v4/registry.json"],
        ["GPU research", "outputs/GPU_RESEARCH_V4.md"],
        ["FULL43", "outputs/FULL43_RESEARCH_V4.md"],
        ["Architecture", "outputs/PRODUCTION_ARCHITECTURE_V4.md"],
    ]
    story.append(data_table(["Доказательство", "Файл"], evidence_rows, [53 * mm, 121 * mm]))
    paragraph(story, "Отчёт описывает только реально выполненные эксперименты. Planned-only кандидаты не смешаны с измеренными результатами.", "CalloutR")

    document.build(story)
    print(json.dumps({"output": str(OUTPUT), "bytes": OUTPUT.stat().st_size}, ensure_ascii=False))


if __name__ == "__main__":
    build()
