from __future__ import annotations

import json
import math
import os
import sqlite3
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from app.core.config import DATABASE_PATH, MODELS_DIR
from app.ml.training import registration_text
from app.ml.v4.runtime import apply_final_policy
from app.services.data_service import human_duration


def load_runtime(models_dir: Path = MODELS_DIR, model_profile: str | None = None) -> dict[str, Any]:
    fallback_category = joblib.load(models_dir / "category.joblib")
    category = fallback_category
    fallback_reason = ""
    profile_fallback_reason = ""
    category_source = "legacy"
    requested_profile = (model_profile or os.getenv("MODEL_PROFILE", "quality")).strip().lower() or "quality"
    if requested_profile not in {"quality", "lite"}:
        raise ValueError("MODEL_PROFILE must be either 'quality' or 'lite'")

    v5_quality_path = models_dir / "v5" / "category.joblib"
    v5_lite_path = models_dir / "v5" / "category_qwen4b_lite.joblib"
    v5_candidates = (
        [(v5_lite_path, "lite"), (v5_quality_path, "quality")]
        if requested_profile == "lite"
        else [(v5_quality_path, "quality")]
    )
    v4_category_path = models_dir / "v4" / "category.joblib"
    active_profile = "legacy"
    category_artifact = models_dir / "category.joblib"
    for v5_category_path, candidate_profile in v5_candidates:
        if not v5_category_path.exists():
            if candidate_profile == requested_profile:
                profile_fallback_reason = f"{candidate_profile} V5 artifact not found"
            continue
        try:
            category = joblib.load(v5_category_path)
            category_source = "v5"
            active_profile = candidate_profile
            category_artifact = v5_category_path
            break
        except Exception as error:
            if candidate_profile == requested_profile:
                profile_fallback_reason = f"{candidate_profile} V5 artifact unavailable: {type(error).__name__}"
            else:
                fallback_reason = f"v5 category unavailable: {type(error).__name__}"
    if category_source != "v5" and v4_category_path.exists():
        try:
            category = joblib.load(v4_category_path)
            category_source = "v4"
            active_profile = "v4"
            category_artifact = v4_category_path
        except Exception as error:
            suffix = f"; v4 category unavailable: {type(error).__name__}"
            fallback_reason = f"{fallback_reason}{suffix}".lstrip("; ")
    routing = joblib.load(models_dir / "routing.joblib")
    retrieval = joblib.load(models_dir / "retrieval.joblib")
    result = {
        "category": category, "routing": routing, "retrieval": retrieval,
        "category_fallback": fallback_category,
        "category_source": category_source,
        "model_profile_requested": requested_profile,
        "model_profile_active": active_profile,
        "category_artifact": str(category_artifact),
        "metadata": {"top15": category["top15"], "threshold": category["threshold"],
                     "routing_threshold": routing.get("threshold")},
    }
    policy_path = models_dir / "v4_runtime.json"
    if category_source != "v5" and policy_path.exists():
        result["v4_policy"] = json.loads(policy_path.read_text(encoding="utf-8"))
    if fallback_reason:
        result["v4_fallback_reason"] = fallback_reason
    if profile_fallback_reason and active_profile != requested_profile:
        result["model_profile_fallback_reason"] = profile_fallback_reason
    return result


def _request_row(payload: dict[str, Any]) -> dict[str, Any]:
    return {key: payload.get(key, "") for key in (
        "description", "registration_date", "user", "service", "component", "request_type",
        "criticality", "urgency", "priority", "service_class", "timezone",
    )}


def _alternatives(model: Any, value: Any, limit: int = 3) -> list[dict[str, Any]]:
    probabilities = model.predict_proba([value])[0]
    classes = model.classes_
    order = np.argsort(probabilities)[::-1][:limit]
    return [{"label": str(classes[index]), "confidence": round(float(probabilities[index]), 4)} for index in order]


def _category_values(bundle: dict[str, Any], row: dict[str, Any], text: str) -> list[Any]:
    if bundle.get("input_contract") == "registration_mapping":
        return [row]
    return [text]


def _category_prediction(
    runtime: dict[str, Any], row: dict[str, Any], text: str
) -> tuple[dict[str, Any], list[dict[str, Any]], str]:
    bundle = runtime["category"]
    try:
        alternatives = _alternatives(bundle["pipeline"], _category_values(bundle, row, text)[0])
        return bundle, alternatives, ""
    except Exception as error:
        fallback = runtime.get("category_fallback")
        if fallback is None or fallback is bundle:
            raise
        alternatives = _alternatives(fallback["pipeline"], _category_values(fallback, row, text)[0])
        return fallback, alternatives, f"{type(error).__name__}: v4 inference failed; v3 fallback used"


def _explain(model: Any, text: str, predicted: str, limit: int = 6) -> list[str]:
    if not hasattr(model, "named_steps"):
        return []
    features = model.named_steps["features"]
    classifier = model.named_steps["classifier"]
    if not hasattr(classifier, "coef_"):
        return []
    vector = features.transform([text])
    names = features.get_feature_names_out()
    class_index = list(classifier.classes_).index(predicted)
    contributions = vector.multiply(classifier.coef_[class_index]).toarray()[0]
    result: list[str] = []
    for index in np.argsort(contributions)[::-1]:
        name = str(names[index])
        if contributions[index] <= 0 or "word__" not in name:
            continue
        token = name.split("word__", 1)[1]
        if "_" in token or len(token) < 3 or token.isdigit():
            continue
        result.append(token)
        if len(result) == limit:
            break
    return result


def retrieve_similar_tickets(runtime: dict[str, Any], payload: dict[str, Any], limit: int = 6) -> dict[str, Any]:
    row = _request_row(payload)
    text = registration_text(row)
    bundle = runtime["retrieval"]
    query = bundle["vectorizer"].transform([text])
    scores = cosine_similarity(query, bundle["matrix"]).ravel()
    if bundle.get("metadata_boost"):
        for index, historical in enumerate(bundle["rows"]):
            scores[index] += 0.08 * bool(row["service"] and row["service"] == historical["service"])
            scores[index] += 0.05 * bool(row["component"] and row["component"] == historical["component"])
            scores[index] += 0.03 * bool(row["request_type"] and row["request_type"] == historical["request_type"])
            scores[index] += 0.02 * bool(row["priority"] and row["priority"] == historical["priority"])
    threshold = float(bundle.get("rejection_threshold", 0.2))
    best_relevance = round(min(1.0, max(0.0, float(scores.max()))) if len(scores) else 0.0, 4)
    if best_relevance < threshold:
        return {
            "rejected": True,
            "reason": "Достаточно похожих исторических обращений не найдено.",
            "threshold": threshold,
            "best_relevance": best_relevance,
            "items": [],
        }
    order = [index for index in np.argsort(scores)[::-1] if float(scores[index]) >= threshold][: max(1, min(limit, 10))]
    result = []
    for index in order:
        historical = bundle["rows"][int(index)]
        result.append({
            "request_id": historical["request_id"], "similarity": round(min(1.0, float(scores[index])), 4),
            "description": historical["description"], "category": historical["category"],
            "final_line": historical["final_line"], "result": historical["result"],
            "overdue": bool(historical["overdue"]), "duration": human_duration(historical["actual_duration_seconds"]),
            "clarifications": historical["clarifications_count"],
        })
    return {
        "rejected": False,
        "reason": "",
        "threshold": threshold,
        "best_relevance": best_relevance,
        "items": result,
    }


def similar_tickets(runtime: dict[str, Any], payload: dict[str, Any], limit: int = 6) -> list[dict[str, Any]]:
    return retrieve_similar_tickets(runtime, payload, limit)["items"]


def _risk_row(connection: sqlite3.Connection, clauses: list[str], values: list[Any]) -> tuple[int, int]:
    where = " WHERE " + " AND ".join(clauses) if clauses else ""
    row = connection.execute(f"SELECT COUNT(*), COALESCE(SUM(overdue), 0) FROM tickets{where}", values).fetchone()
    return int(row[0]), int(row[1])


def historical_sla_risk(
    payload: dict[str, Any], category: str | None, routing: str | None, similar: list[dict[str, Any]], database: Path = DATABASE_PATH
) -> dict[str, Any]:
    connection = sqlite3.connect(database)
    global_n, global_overdue = _risk_row(connection, [], [])
    global_rate = global_overdue / global_n if global_n else 0.0
    candidates: list[tuple[str, list[str], list[Any], int]] = []
    strong_ids = [item["request_id"] for item in similar if item["similarity"] >= 0.45]
    if len(strong_ids) >= 5:
        candidates.append(("достаточно похожие обращения", [f"request_id IN ({','.join('?' for _ in strong_ids)})"], strong_ids, 5))
    service, priority = payload.get("service"), payload.get("priority")
    if category and service and priority and routing:
        candidates.append(("категория + услуга + приоритет + линия", ["category=?", "service=?", "priority=?", "final_line=?"], [category, service, priority, routing], 20))
    if category and service:
        candidates.append(("категория + услуга", ["category=?", "service=?"], [category, service], 20))
    if category:
        candidates.append(("категория", ["category=?"], [category], 30))
    candidates.append(("общий уровень", [], [], 1))
    basis, sample, overdue = "общий уровень", global_n, global_overdue
    for name, clauses, values, minimum in candidates:
        n, late = _risk_row(connection, clauses, values)
        if n >= minimum:
            basis, sample, overdue = name, n, late
            break
    connection.close()
    prior_strength = 30
    risk = (overdue + prior_strength * global_rate) / (sample + prior_strength) if sample + prior_strength else 0.0
    raw_rate = overdue / sample if sample else 0.0
    effective_n = sample + prior_strength
    standard_error = math.sqrt(max(risk * (1 - risk), 0) / effective_n) if effective_n else 0.0
    interval = [max(0.0, risk - 1.96 * standard_error), min(1.0, risk + 1.96 * standard_error)]
    reliability = "высокая" if sample >= 100 else "средняя" if sample >= 30 else "низкая"
    level = "Повышенный" if risk >= max(0.05, global_rate * 2) else "Низкий"
    return {
        "risk": round(float(risk), 4), "raw_rate": round(float(raw_rate), 4), "sample_size": sample,
        "support_n": sample, "basis": basis, "segment": basis, "global_rate": round(float(global_rate), 4),
        "reliability": reliability, "interval_95": [round(value, 4) for value in interval], "level": level,
        "explanation": "Beta-сглаженная историческая доля с иерархическим fallback; это индикатор, не гарантированный прогноз.",
    }


def _review_reason(alternatives: list[dict[str, Any]], threshold: float, margin: float = 0.0) -> str:
    top = alternatives[0]
    second = alternatives[1] if len(alternatives) > 1 else {"label": "—", "confidence": 0.0}
    actual_margin = top["confidence"] - second["confidence"]
    if top["confidence"] < threshold:
        return f"Максимальная уверенность {top['confidence']:.1%} ниже порога {threshold:.1%}."
    if actual_margin < margin:
        return f"Лучшие варианты близки: {top['label']} {top['confidence']:.1%} и {second['label']} {second['confidence']:.1%}."
    return "Обращение отличается от уверенно распознаваемых примеров."


def _historical_route_prior(
    database: Path,
    category_weights: list[tuple[str, float]],
    labels: list[str],
) -> np.ndarray | None:
    if not category_weights:
        return None
    try:
        connection = sqlite3.connect(database)
        columns = {row[1] for row in connection.execute("PRAGMA table_info(tickets)")}
        source_filter = " AND source_hash <> 'operator-review'" if "source_hash" in columns else ""
        total = np.zeros(len(labels), dtype=float)
        total_weight = 0.0
        for category, category_weight in category_weights:
            if category_weight <= 0:
                continue
            rows = connection.execute(
                "SELECT final_line, COUNT(*) FROM tickets "
                f"WHERE category=? AND final_line IN ('(1 линия)', '(2 линия)', '(3 линия)'){source_filter} "
                "GROUP BY final_line",
                (category,),
            ).fetchall()
            counts = {str(line): int(count) for line, count in rows}
            support = sum(counts.values())
            if support < 3:
                continue
            smoothed = np.asarray([counts.get(label, 0) + 1.0 for label in labels], dtype=float)
            smoothed /= smoothed.sum()
            total += float(category_weight) * smoothed
            total_weight += float(category_weight)
        connection.close()
        if total_weight <= 0:
            return None
        return total / total_weight
    except sqlite3.Error:
        return None


def route_ticket(
    runtime: dict[str, Any],
    payload: dict[str, Any],
    *,
    category_alternatives: list[dict[str, Any]] | None = None,
    confirmed_category: str | None = None,
    database: Path = DATABASE_PATH,
) -> dict[str, Any]:
    row = _request_row(payload)
    text = registration_text(row)
    routing_bundle = runtime["routing"]
    routing_model = routing_bundle["pipeline"]
    labels = [str(item) for item in routing_model.classes_]
    base_probabilities = np.asarray(routing_model.predict_proba([text])[0], dtype=float)

    if confirmed_category:
        mode = "confirmed_category"
        category_weights = [(confirmed_category, 1.0)]
    else:
        mode = "category_topk" if category_alternatives else "registration_only"
        category_weights = [
            (str(item.get("label") or ""), float(item.get("confidence") or 0.0))
            for item in (category_alternatives or [])[:3]
            if item.get("label")
        ]
    prior = _historical_route_prior(database, category_weights, labels)
    probabilities = base_probabilities
    if prior is not None:
        probabilities = 0.45 * base_probabilities + 0.55 * prior
    probabilities = probabilities / probabilities.sum()
    order = np.argsort(probabilities)[::-1]
    alternatives = [
        {"label": labels[index], "confidence": round(float(probabilities[index]), 4)}
        for index in order[:3]
    ]
    best = alternatives[0]
    threshold = float(routing_bundle.get("threshold", 0.55))
    accepted = best["confidence"] >= threshold
    return {
        "label": best["label"],
        "confidence": best["confidence"],
        "accepted": accepted,
        "review_required": not accepted,
        "threshold": threshold,
        "review_reason": "" if accepted else _review_reason(alternatives, threshold),
        "alternatives": alternatives,
        "signals": _explain(routing_bundle.get("explanation_pipeline"), text, best["label"], 4),
        "explanation": (
            "Маршрут рассчитан по регистрационным полям и подтверждённой категории."
            if confirmed_category
            else "Маршрут рассчитан по регистрационным полям и TOP-K категорий модели."
        ) + " 4-я линия требует ручной оценки из-за одного исторического примера.",
        "mode": mode,
        "category_prior_used": prior is not None,
    }


def analyze_ticket(runtime: dict[str, Any], payload: dict[str, Any], database: Path = DATABASE_PATH) -> dict[str, Any]:
    row = _request_row(payload)
    text = registration_text(row)
    category_bundle, category_alternatives, category_fallback_reason = _category_prediction(
        runtime, row, text
    )
    category_best = category_alternatives[0]
    category_threshold = float(category_bundle["threshold"])
    margin_threshold = float(category_bundle.get("margin_threshold", 0.0))
    actual_margin = category_best["confidence"] - category_alternatives[1]["confidence"]
    category_accepted = category_best["confidence"] >= category_threshold and actual_margin >= margin_threshold

    routing = route_ticket(
        runtime,
        payload,
        category_alternatives=category_alternatives,
        database=database,
    )
    retrieval = retrieve_similar_tickets(runtime, payload)
    similar = retrieval["items"]
    sla_category = category_best["label"] if category_accepted else None
    sla_routing = routing["label"] if routing["accepted"] else None
    result = {
        "category": {
            "label": category_best["label"], "confidence": category_best["confidence"],
            "accepted": category_accepted, "review_required": not category_accepted,
            "threshold": category_threshold, "margin_threshold": margin_threshold,
            "review_reason": "" if category_accepted else _review_reason(category_alternatives, category_threshold, margin_threshold),
            "alternatives": category_alternatives,
            "signals": _explain(category_bundle.get("explanation_pipeline"), text, category_best["label"]),
            "explanation": "Положительные слова и сочетания из локальной линейной модели; уверенность откалибрована на grouped OOF данных.",
        },
        "routing": routing,
        "sla_risk": historical_sla_risk(payload, sla_category, sla_routing, similar, database),
        "retrieval": {key: value for key, value in retrieval.items() if key != "items"},
        "similar": similar,
    }
    if "v4_policy" in runtime:
        result = apply_final_policy(result, runtime["v4_policy"])
    result["model_provenance"] = dict(category_bundle.get("metadata", {}))
    if category_fallback_reason:
        result["model_provenance"].update({
            "fallback": True,
            "fallback_reason": category_fallback_reason,
        })
    return result
