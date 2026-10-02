from __future__ import annotations

import json
import re
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np
import yaml
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TOKEN_RE = re.compile(r"(?u)\b[\w-]{2,}\b")
ABBR_RE = re.compile(r"(?<![A-Za-zА-ЯЁа-яё])(?:[A-ZА-ЯЁ0-9]{2,}|[A-ZА-ЯЁ]{1,5}[-/][A-ZА-ЯЁ0-9]{1,8})(?![A-Za-zА-ЯЁа-яё])")
GENERIC = {
    "добрый", "день", "просьба", "пожалуйста", "проблема", "ошибка", "обращение",
    "письма", "текст", "тема", "номер", "телефон", "клиент", "работает", "работы",
}
OFFICIAL_CONTEXT: dict[str, list[dict[str, Any]]] = {
    "Проблема с QR-код(подключение/отключение)": [{
        "url": "https://info.pochta.ru/support/office-services/pep",
        "terminology": ["QR-код", "простая электронная подпись", "мобильное приложение", "готово к вручению"],
        "boundary_note": "QR-код получения связан с подключённой электронной подписью; общий сбой входа без QR-сценария относится к авторизации.",
    }],
    "Проблема с авторизацией/ЭЗП/Бонусами/Доверенностями": [{
        "url": "https://info.pochta.ru/support/office-services/pep",
        "terminology": ["электронная подпись", "учётная запись Госуслуг", "электронная доверенность"],
        "boundary_note": "Отделять подключение подписи/доверенности от показа QR-кода уже подключённому пользователю.",
    }],
    "Проблемы в работе Препост/PrePost": [{
        "url": "https://info.pochta.ru/support/business/prepost",
        "terminology": ["PrePost", "офлайн", "партионные отправления", "список ф. 103", "тарификатор"],
        "boundary_note": "PrePost — офлайн-подготовка партионной корреспонденции; не смешивать с импортом списка в Priem.pochta.ru.",
    }],
    "Установка Препост/PrePost в ОПС": [{
        "url": "https://info.pochta.ru/support/business/prepost",
        "terminology": ["установочный файл", "Windows", "офлайн-тарификатор", "PrePost"],
        "boundary_note": "Категория установки требует инсталляционного сценария, а не ошибки уже работающей программы.",
    }],
    "Импорт списков из архива ф.103 (zip-архив)": [{
        "url": "https://info.pochta.ru/support/business/prepost",
        "terminology": ["список ф. 103", "партионные отправления", "архив", "офлайн"],
        "boundary_note": "Ключевой объект — архив/список ф.103; PrePost может быть источником, но не обязательно местом сбоя.",
    }],
    "Отслеживание отправлений": [{
        "url": "https://info.pochta.ru/support/post-rules/receiving-sending",
        "terminology": ["трек-номер", "статус отслеживания", "готово к вручению", "извещение"],
        "boundary_note": "Отделять отсутствие/ошибку статуса от физической проблемы выдачи или доставки в ОПС.",
    }],
    "Тарификация посылок": [{
        "url": "https://tariff.pochta.ru/post-calculator-api.pdf",
        "terminology": ["тариф", "расчёт стоимости", "вес", "категория отправителя", "договорной тариф"],
        "boundary_note": "Нулевой real-train класс: определение основано только на названии таксономии и официальной терминологии расчёта тарифов; синтетика помечается SYNTHETIC_ONLY.",
    }],
}


def _quantile(values: list[int], q: float) -> int:
    return int(round(float(np.quantile(np.asarray(values, dtype=float), q)))) if values else 0


def _top_metadata(rows: list[dict[str, Any]], key: str, n: int = 3) -> list[dict[str, Any]]:
    counts = Counter(str(row.get(key) or "").strip() for row in rows)
    counts.pop("", None)
    return [{"value": value, "count": count} for value, count in counts.most_common(n)]


def main() -> None:
    pointer = json.loads((PROJECT_ROOT / "artifacts/gpu_research_v4/protocol.json").read_text(encoding="utf-8"))
    protocol_path = Path(pointer["protocol_path"])
    if not protocol_path.is_absolute():
        protocol_path = PROJECT_ROOT / protocol_path
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    fold = next(item for item in protocol["folds"] if int(item["repeat"]) == 0 and int(item["fold"]) == 0)

    connection = sqlite3.connect(PROJECT_ROOT / "data/processed/posttech.db")
    connection.row_factory = sqlite3.Row
    by_id = {str(row["request_id"]): dict(row) for row in connection.execute("SELECT * FROM tickets")}
    train_rows = [by_id[str(request_id)] for request_id in fold["train_request_ids"]]
    labels = [str(label) for label in protocol["labels"]]
    texts = [str(row.get("description") or "") for row in train_rows]
    y = [str(row["category"]) for row in train_rows]

    # Distinctive vocabulary is fitted only on the frozen real TRAIN partition.
    word_vectorizer = TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=1, max_df=0.92, max_features=30_000)
    word_matrix = word_vectorizer.fit_transform(texts)
    feature_names = np.asarray(word_vectorizer.get_feature_names_out())

    # Character centroids expose empirically close categories without looking at validation/test.
    char_vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1, max_features=40_000)
    char_matrix = char_vectorizer.fit_transform(texts)
    indices_by_label = {label: np.asarray([i for i, value in enumerate(y) if value == label], dtype=int) for label in labels}
    centroid_labels = [label for label in labels if len(indices_by_label[label])]
    centroids = np.vstack([np.asarray(char_matrix[indices_by_label[label]].mean(axis=0)) for label in centroid_labels])
    similarities = cosine_similarity(char_matrix, centroids)
    confusions: dict[str, Counter[str]] = defaultdict(Counter)
    for row_index, own_label in enumerate(y):
        order = np.argsort(similarities[row_index])[::-1]
        for candidate_index in order:
            candidate = centroid_labels[int(candidate_index)]
            if candidate != own_label:
                confusions[own_label][candidate] += 1
                break

    profiles: dict[str, Any] = {}
    definitions_path = PROJECT_ROOT / "data/category_definitions_v4.yaml"
    existing = yaml.safe_load(definitions_path.read_text(encoding="utf-8"))
    existing_by_label = {str(item["category"]): item for item in existing}
    enriched: list[dict[str, Any]] = []

    for label in labels:
        indices = indices_by_label[label]
        rows = [train_rows[int(index)] for index in indices]
        class_texts = [texts[int(index)] for index in indices]
        char_lengths = [len(text) for text in class_texts]
        token_lengths = [len(TOKEN_RE.findall(text)) for text in class_texts]
        abbreviations = Counter(match.group(0) for text in class_texts for match in ABBR_RE.finditer(text))

        if len(indices):
            class_scores = np.asarray(word_matrix[indices].mean(axis=0)).ravel()
            ranked = np.argsort(class_scores)[::-1]
            distinctive_terms = [
                str(feature_names[int(index)])
                for index in ranked
                if str(feature_names[int(index)]).split()[0] not in GENERIC
            ][:18]
        else:
            distinctive_terms = []

        neighbours = [name for name, _ in confusions[label].most_common(5)]
        metadata = {
            key: _top_metadata(rows, key)
            for key in ("service", "component", "request_type", "criticality", "priority", "service_class")
        }
        style = {
            "characters": {"p10": _quantile(char_lengths, 0.10), "median": int(median(char_lengths)) if char_lengths else 0, "p90": _quantile(char_lengths, 0.90)},
            "tokens": {"p10": _quantile(token_lengths, 0.10), "median": int(median(token_lengths)) if token_lengths else 0, "p90": _quantile(token_lengths, 0.90)},
            "share_with_email_template": round(sum("Тема письма" in text or "Текст письма" in text for text in class_texts) / max(1, len(class_texts)), 4),
            "share_with_ops_template": round(sum("Индекс ОПС" in text or "Номер окна" in text for text in class_texts) / max(1, len(class_texts)), 4),
            "share_with_url": round(sum("http" in text.lower() for text in class_texts) / max(1, len(class_texts)), 4),
            "share_with_uppercase_tokens": round(sum(bool(ABBR_RE.search(text)) for text in class_texts) / max(1, len(class_texts)), 4),
        }
        profile = {
            "real_train_support": len(rows),
            "language": "ru (смешанные служебные латинские токены и аббревиатуры)",
            "style": style,
            "distinctive_terms": distinctive_terms,
            "common_abbreviations": [name for name, _ in abbreviations.most_common(15)],
            "recurring_entities": metadata,
            "empirical_confusion_categories": neighbours,
            "ambiguity_risk": "very_high" if len(rows) <= 5 else "high" if len(rows) <= 15 else "medium",
        }
        profiles[label] = profile

        item = dict(existing_by_label.get(label, {"category": label}))
        services = [entry["value"] for entry in metadata["service"][:2]]
        components = [entry["value"] for entry in metadata["component"][:2]]
        context = ", ".join([*services, *components]) or "контекст определяется основным предметом обращения"
        terms = distinctive_terms[:8]
        item["definition"] = (
            f"Основной предмет обращения — «{label}». В реальном TRAIN характерный контекст: {context}. "
            f"Различающие формулировки/термины: {', '.join(terms) if terms else 'недостаточно real-train примеров; требуется осторожная семантическая проверка'}."
        )
        item["typical_symptoms"] = terms
        item["belongs"] = [
            f"Основная пользовательская проблема или требуемая операция непосредственно относится к «{label}».",
            f"Реальный TRAIN-контекст: {context}.",
        ]
        item["does_not_belong"] = [
            f"Не относить сюда, когда «{label}» упомянуто лишь фоном, а действие/сбой относится к другому объекту.",
            *( [f"Особенно проверять границу с: {', '.join(neighbours)}."] if neighbours else [] ),
        ]
        item["common_terminology"] = terms
        item["common_abbreviations"] = profile["common_abbreviations"]
        item["confusable_categories"] = neighbours
        item["real_train_support"] = len(rows)
        item["real_train_style_profile"] = style
        item["recurring_entities"] = metadata
        item["ambiguity_risk"] = profile["ambiguity_risk"]
        item["definition_provenance"] = {
            "dataset_scope": "frozen_v4_repeat0_fold0_real_train_only",
            "protocol_hash": pointer["split_sha256"],
            "documentation": ["hackathon task statement", "dataset field semantics"],
            "internet_text_used_as_training_data": False,
        }
        item["official_reference_context"] = OFFICIAL_CONTEXT.get(label, [])
        enriched.append(item)

    output_dir = PROJECT_ROOT / "artifacts/gpu_research_v4/synthetic"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "real_train_category_profiles.json").write_text(
        json.dumps({"scope": "REAL_TRAIN_ONLY", "profiles": profiles}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    definitions_path.write_text(yaml.safe_dump(enriched, allow_unicode=True, sort_keys=False, width=120), encoding="utf-8")
    print(json.dumps({"categories": len(labels), "train_rows": len(train_rows), "definitions": str(definitions_path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
