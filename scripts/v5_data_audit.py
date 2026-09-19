from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import statistics
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any
from zipfile import ZipFile


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = PROJECT_ROOT / "data" / "raw" / "Обращения_1931.xlsx"
ARTIFACT_ROOT = PROJECT_ROOT / "artifacts" / "gpu_research_v5"
OUTPUT_ROOT = PROJECT_ROOT / "outputs"

CATEGORY_TARGET = "Вид запроса"
ROUTING_TARGET = "Кем решен (группа)"

CORE_REGISTRATION_FIELDS = (
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
)

UNCERTAIN_REGISTRATION_FIELDS = (
    "Норматив ное время обработки (SLA)",
    "Плановое время выполнения",
    "Крайний срок обработки",
)

POST_RESOLUTION_FIELDS = (
    "Статус",
    "Фактическое время выполнения",
    "Фактическая длительность выполнения запроса (SLA)",
    "Просрочен?*",
    "Время входа в статус",
    "Дата последнего изменения",
    "Суммарное время реакции 1 линии",
    "Суммарное время работы 1 линии",
    "Суммарное время реакции 2 линии",
    "Суммарное время работы 2 линии",
    "Суммарное время реакции 3 линии",
    "Суммарное время работы 3 линии",
    "Суммарное время реакции 4 линии",
    "Суммарное время работы 4 линии",
    "Суммарное время уточнений",
    "Результат работ",
    "Количество уточнений",
)

DATETIME_FIELDS = {
    "Дата регистрации",
    "Плановое время выполнения",
    "Крайний срок обработки",
    "Фактическое время выполнения",
    "Время входа в статус",
    "Дата последнего изменения",
}

DURATION_FIELDS = {
    "Норматив ное время обработки (SLA)",
    "Фактическая длительность выполнения запроса (SLA)",
    "Суммарное время реакции 1 линии",
    "Суммарное время работы 1 линии",
    "Суммарное время реакции 2 линии",
    "Суммарное время работы 2 линии",
    "Суммарное время реакции 3 линии",
    "Суммарное время работы 3 линии",
    "Суммарное время реакции 4 линии",
    "Суммарное время работы 4 линии",
    "Суммарное время уточнений",
}

NS_MAIN = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
NS_REL = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
NS_PKG_REL = "http://schemas.openxmlformats.org/package/2006/relationships"


def _column_index(reference: str) -> int:
    match = re.match(r"([A-Z]+)", reference)
    if not match:
        raise ValueError(f"Invalid Excel reference: {reference}")
    result = 0
    for char in match.group(1):
        result = result * 26 + (ord(char) - ord("A") + 1)
    return result - 1


def _shared_strings(root: ET.Element) -> list[str]:
    values: list[str] = []
    for item in root.findall(f"{{{NS_MAIN}}}si"):
        values.append("".join(node.text or "" for node in item.iter(f"{{{NS_MAIN}}}t")))
    return values


def _cell_value(cell: ET.Element, shared: list[str]) -> tuple[Any, str]:
    cell_type = cell.attrib.get("t")
    value_node = cell.find(f"{{{NS_MAIN}}}v")
    if cell_type == "inlineStr":
        text = "".join(node.text or "" for node in cell.iter(f"{{{NS_MAIN}}}t"))
        return (text or None), "inline_string"
    if value_node is None or value_node.text is None:
        return None, "blank"
    raw = value_node.text
    if cell_type == "s":
        return shared[int(raw)], "shared_string"
    if cell_type == "b":
        return raw == "1", "boolean"
    if cell_type in {"str", "e"}:
        return raw, "string" if cell_type == "str" else "error"
    try:
        numeric = float(raw)
    except ValueError:
        return raw, "string"
    if numeric.is_integer():
        return int(numeric), "number"
    return numeric, "number"


def read_xlsx_rows(path: Path) -> tuple[str, list[str], list[dict[str, Any]], dict[str, dict[str, int]]]:
    """Read the first XLSX sheet with the Python standard library only.

    V5 intentionally avoids depending on the historical openpyxl importer for the
    initial audit so the dataset contract is independently recalculated.
    """

    path = Path(path)
    with ZipFile(path) as archive:
        workbook = ET.fromstring(archive.read("xl/workbook.xml"))
        first_sheet = workbook.find(f".//{{{NS_MAIN}}}sheet")
        if first_sheet is None:
            raise ValueError("XLSX contains no worksheets")
        sheet_name = first_sheet.attrib["name"]
        relation_id = first_sheet.attrib[f"{{{NS_REL}}}id"]

        relationships = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
        target = None
        for relation in relationships.findall(f"{{{NS_PKG_REL}}}Relationship"):
            if relation.attrib.get("Id") == relation_id:
                target = relation.attrib["Target"]
                break
        if target is None:
            raise ValueError(f"Cannot resolve worksheet relation {relation_id}")
        sheet_path = target.lstrip("/")
        if not sheet_path.startswith("xl/"):
            sheet_path = f"xl/{sheet_path}"

        shared: list[str] = []
        if "xl/sharedStrings.xml" in archive.namelist():
            shared = _shared_strings(ET.fromstring(archive.read("xl/sharedStrings.xml")))
        sheet = ET.fromstring(archive.read(sheet_path))

    xml_rows = sheet.findall(f".//{{{NS_MAIN}}}sheetData/{{{NS_MAIN}}}row")
    if not xml_rows:
        raise ValueError("Worksheet is empty")

    header_cells: dict[int, Any] = {}
    for cell in xml_rows[0].findall(f"{{{NS_MAIN}}}c"):
        value, _ = _cell_value(cell, shared)
        if value not in {None, ""}:
            header_cells[_column_index(cell.attrib["r"])] = str(value).strip()
    if not header_cells:
        raise ValueError("Worksheet has no named header cells")
    last_header_index = max(header_cells)
    headers = [str(header_cells.get(index, "")).strip() for index in range(last_header_index + 1)]
    if any(not header for header in headers):
        raise ValueError("Named header range contains an empty internal column")

    rows: list[dict[str, Any]] = []
    storage: dict[str, Counter[str]] = {header: Counter() for header in headers}
    for xml_row in xml_rows[1:]:
        values: list[Any] = [None] * len(headers)
        types: list[str] = ["blank"] * len(headers)
        for cell in xml_row.findall(f"{{{NS_MAIN}}}c"):
            index = _column_index(cell.attrib["r"])
            if index >= len(headers):
                continue
            values[index], types[index] = _cell_value(cell, shared)
        if not any(value not in {None, ""} for value in values):
            continue
        row = dict(zip(headers, values, strict=True))
        rows.append(row)
        for header, storage_type in zip(headers, types, strict=True):
            storage[header][storage_type] += 1

    return sheet_name, headers, rows, {header: dict(counter) for header, counter in storage.items()}


def _semantic_type(name: str) -> str:
    if name == "Номер запроса":
        return "identifier"
    if name in DATETIME_FIELDS:
        return "datetime"
    if name in DURATION_FIELDS:
        return "duration"
    if name == "Описание 2":
        return "free_text"
    if name in {CATEGORY_TARGET, ROUTING_TARGET}:
        return "target_categorical"
    if name == "Количество уточнений":
        return "integer"
    return "categorical"


def _policy_for_field(name: str) -> dict[str, Any]:
    if name == "Номер запроса":
        return {
            "available_at_registration": "yes",
            "optional": False,
            "safe_for_category": False,
            "safe_for_routing": False,
            "is_target": False,
            "post_resolution_or_leakage": False,
            "predictive_role": "identity_only",
            "routing_usage": "identity_tracking_only",
            "evidence": "Request ID exists at registration but is an identity/tracking key, not a predictive signal.",
        }
    if name in CORE_REGISTRATION_FIELDS:
        note = "Listed by the V5.1 brief as a likely registration-time input."
        if name == "Пользователь":
            note += " High cardinality/generalization must be benchmarked with and without this field."
        if name == "Описание 2":
            note += " It may be empty/short in production, so metadata fallback and low-information warnings are required."
        return {
            "available_at_registration": "yes",
            "optional": name != "Дата регистрации",
            "safe_for_category": True,
            "safe_for_routing": True,
            "is_target": False,
            "post_resolution_or_leakage": False,
            "predictive_role": "core_registration_feature",
            "routing_usage": "direct_registration_feature",
            "evidence": note,
        }
    if name == CATEGORY_TARGET:
        return {
            "available_at_registration": "no",
            "optional": False,
            "safe_for_category": False,
            "safe_for_routing": False,
            "is_target": True,
            "post_resolution_or_leakage": False,
            "predictive_role": "category_target",
            "routing_usage": "confirmed_only_or_oof_prediction",
            "evidence": "Category is the supervised target. Pre-confirmation routing may use OOF predicted category probabilities; post-confirmation routing may use the operator-confirmed category.",
        }
    if name == ROUTING_TARGET:
        return {
            "available_at_registration": "no",
            "optional": False,
            "safe_for_category": False,
            "safe_for_routing": False,
            "is_target": True,
            "post_resolution_or_leakage": False,
            "predictive_role": "routing_target",
            "routing_usage": "supervised_target",
            "evidence": "Routing supervised target.",
        }
    if name in POST_RESOLUTION_FIELDS:
        return {
            "available_at_registration": "no",
            "optional": True,
            "safe_for_category": False,
            "safe_for_routing": False,
            "is_target": False,
            "post_resolution_or_leakage": True,
            "predictive_role": "forbidden_post_resolution",
            "routing_usage": "forbidden_feature",
            "evidence": "Produced or finalized after ticket processing; using it at first recommendation time would leak future information.",
        }
    if name in UNCERTAIN_REGISTRATION_FIELDS:
        return {
            "available_at_registration": "uncertain",
            "optional": True,
            "safe_for_category": False,
            "safe_for_routing": False,
            "is_target": False,
            "post_resolution_or_leakage": False,
            "predictive_role": "conditional_registration_feature",
            "routing_usage": "exclude_until_availability_is_proven",
            "evidence": "Potential planning/registration field; product-time availability was not proven, so it is excluded from the core model contract.",
        }
    return {
        "available_at_registration": "unknown",
        "optional": True,
        "safe_for_category": False,
        "safe_for_routing": False,
        "is_target": False,
        "post_resolution_or_leakage": False,
        "predictive_role": "excluded_or_unknown",
        "routing_usage": "excluded_or_unknown",
        "evidence": "Not part of the core V5 registration-time contract.",
    }


def _field_contract(header: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    values = [row[header] for row in rows]
    nonmissing = [value for value in values if value not in {None, ""}]
    missing = len(values) - len(nonmissing)
    cardinality = len({_normalize_text(value) for value in nonmissing})
    policy = _policy_for_field(header)
    return {
        "name": header,
        "semantic_type": _semantic_type(header),
        "missing_count": missing,
        "missing_percentage": round(missing * 100.0 / len(values), 6) if values else None,
        "cardinality_nonmissing": cardinality,
        **policy,
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(str(value).split())


def _percentile(values: list[int], quantile: float) -> int | None:
    if not values:
        return None
    sorted_values = sorted(values)
    position = int((len(sorted_values) - 1) * quantile)
    return sorted_values[position]


def _description_stats(values: list[Any]) -> dict[str, Any]:
    texts = [_normalize_text(value) for value in values]
    lengths = [len(text) for text in texts]
    tokens = [len(text.split()) for text in texts]
    return {
        "nonempty": sum(bool(text) for text in texts),
        "empty": sum(not text for text in texts),
        "characters": {
            "min": min(lengths) if lengths else None,
            "median": statistics.median(lengths) if lengths else None,
            "p90": _percentile(lengths, 0.90),
            "p95": _percentile(lengths, 0.95),
            "p99": _percentile(lengths, 0.99),
            "max": max(lengths) if lengths else None,
        },
        "whitespace_tokens": {
            "min": min(tokens) if tokens else None,
            "median": statistics.median(tokens) if tokens else None,
            "p90": _percentile(tokens, 0.90),
            "p95": _percentile(tokens, 0.95),
            "p99": _percentile(tokens, 0.99),
            "max": max(tokens) if tokens else None,
        },
    }


def _category_line_distribution(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_category: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        category = _normalize_text(row[CATEGORY_TARGET])
        line = _normalize_text(row[ROUTING_TARGET])
        by_category[category][line] += 1
    result = []
    for category in sorted(by_category):
        counts = by_category[category]
        total = sum(counts.values())
        result.append(
            {
                "category": category,
                "total": total,
                "line_counts": dict(counts),
                "line_probabilities": {
                    line: round(count / total, 6) for line, count in counts.items()
                },
            }
        )
    return result


def build_data_contract(path: Path) -> dict[str, Any]:
    path = Path(path)
    sheet_name, headers, rows, source_storage = read_xlsx_rows(path)
    if len(rows) != 1931:
        raise ValueError(f"Expected 1931 data rows, found {len(rows)}")
    duplicate_headers = [name for name, count in Counter(headers).items() if count > 1]
    if duplicate_headers:
        raise ValueError(f"Duplicate XLSX headers: {duplicate_headers}")
    if "Номер запроса" not in headers:
        raise ValueError("Номер запроса column is missing")
    request_ids = [_normalize_text(row["Номер запроса"]) for row in rows]
    if any(not value for value in request_ids):
        raise ValueError("Request ID contains empty values")
    if len(set(request_ids)) != len(request_ids):
        raise ValueError("Request ID must be unique")

    category_counts = Counter(_normalize_text(row[CATEGORY_TARGET]) for row in rows)
    if "" in category_counts:
        raise ValueError("Category target contains missing values")
    routing_counts = Counter(_normalize_text(row[ROUTING_TARGET]) for row in rows)
    if "" in routing_counts:
        raise ValueError("Routing target contains missing values")

    top_categories = category_counts.most_common(15)
    fields = []
    for header in headers:
        field = _field_contract(header, rows)
        field["source_storage_types"] = source_storage[header]
        fields.append(field)

    category_distribution = [
        {"name": name, "count": count} for name, count in category_counts.most_common()
    ]
    top15 = [{"name": name, "count": count} for name, count in top_categories]
    routing_distribution = [
        {"name": name, "count": count} for name, count in routing_counts.most_common()
    ]

    contract = {
        "dataset": {
            "path": str(path.resolve()),
            "sha256": _sha256_file(path),
            "sheet": sheet_name,
            "records": len(rows),
            "source_columns": len(headers),
        },
        "core_category_features": list(CORE_REGISTRATION_FIELDS),
        "core_routing_features": list(CORE_REGISTRATION_FIELDS),
        "conditional_optional_fields": list(UNCERTAIN_REGISTRATION_FIELDS),
        "fields": fields,
        "category_summary": {
            "distinct_categories": len(category_distribution),
            "top15_records": sum(item["count"] for item in top15),
            "top15": top15,
            "all_categories": category_distribution,
        },
        "routing_summary": {
            "distinct_lines": len(routing_distribution),
            "distribution": routing_distribution,
            "category_line_distribution": _category_line_distribution(rows),
        },
        "description_length": _description_stats([row.get("Описание 2") for row in rows]),
        "rules": {
            "request_id_predictive": False,
            "true_category_allowed_for_preconfirmation_routing": False,
            "preconfirmation_category_signal": "OOF predicted category probabilities / TOP-K only",
            "confirmed_category_allowed_for_postconfirmation_routing": True,
            "post_resolution_features_allowed": False,
            "synthetic_validation_allowed": False,
        },
    }
    _validate_contract(contract)
    return contract


def _validate_contract(contract: dict[str, Any]) -> None:
    by_name = {field["name"]: field for field in contract["fields"]}
    unknown = set(by_name) - (
        {"Номер запроса", CATEGORY_TARGET, ROUTING_TARGET}
        | set(CORE_REGISTRATION_FIELDS)
        | set(UNCERTAIN_REGISTRATION_FIELDS)
        | set(POST_RESOLUTION_FIELDS)
    )
    if unknown:
        raise ValueError(f"Unclassified source fields: {sorted(unknown)}")
    forbidden = {
        field["name"]
        for field in contract["fields"]
        if field["post_resolution_or_leakage"] or field["is_target"]
    }
    for feature_key in ("core_category_features", "core_routing_features"):
        overlap = forbidden.intersection(contract[feature_key])
        if overlap:
            raise ValueError(f"Leakage fields found in {feature_key}: {sorted(overlap)}")


def _md_escape(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def write_outputs(contract: dict[str, Any]) -> None:
    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    (ARTIFACT_ROOT / "data_contract.json").write_text(
        json.dumps(contract, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    summary = {
        "dataset": contract["dataset"],
        "category_summary": contract["category_summary"],
        "routing_summary": contract["routing_summary"],
        "description_length": contract["description_length"],
    }
    (ARTIFACT_ROOT / "dataset_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Recalculate the PostTech Radar V5.1 XLSX data contract")
    parser.add_argument("dataset", nargs="?", type=Path, default=DEFAULT_DATASET)
    args = parser.parse_args()
    contract = build_data_contract(args.dataset)
    write_outputs(contract)
    print(json.dumps({"records": contract["dataset"]["records"], "sha256": contract["dataset"]["sha256"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
