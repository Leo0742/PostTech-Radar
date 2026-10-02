from __future__ import annotations

import hashlib
import random
from typing import Any, Mapping


OPTIONAL_FIELDS = (
    "description",
    "user",
    "service",
    "component",
    "request_type",
    "criticality",
    "urgency",
    "priority",
    "service_class",
    "timezone",
)

SCENARIOS = (
    "full",
    "description_only",
    "metadata_only",
    "no_component",
    "no_service",
    "no_request_type",
    "no_priority_criticality",
    "random_dropout_10",
    "random_dropout_30",
    "random_dropout_50",
    "empirical_missingness",
)

# Recalculated from the supplied 1,931-ticket XLSX. Most registration fields are
# complete in the historical export; component is the material observed gap.
EMPIRICAL_MISSINGNESS = {
    "description": 0.0,
    "user": 0.0,
    "service": 0.0,
    "component": 619 / 1931,
    "request_type": 0.0,
    "criticality": 0.0,
    "urgency": 0.0,
    "priority": 0.0,
    "service_class": 0.0,
    "timezone": 0.0,
}


def _stable_rng(seed: int, scenario: str, row_key: str) -> random.Random:
    digest = hashlib.sha256(f"{seed}|{scenario}|{row_key}".encode("utf-8")).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


def _row_key(values: Mapping[str, Any]) -> str:
    parts = []
    for field in OPTIONAL_FIELDS:
        value = values.get(field)
        parts.append(f"{field}={'' if value is None else value}")
    return "|".join(parts)


def _drop_random_fields(
    result: dict[str, Any], *, probability: float, seed: int, scenario: str
) -> dict[str, Any]:
    rng = _stable_rng(seed, scenario, _row_key(result))
    for field in OPTIONAL_FIELDS:
        if rng.random() < probability:
            result[field] = None
    return result


def apply_missingness_scenario(
    values: Mapping[str, Any], scenario: str, *, seed: int = 20260917
) -> dict[str, Any]:
    if scenario not in SCENARIOS:
        raise ValueError(f"Unknown missingness scenario: {scenario}")

    result = dict(values)
    for field in OPTIONAL_FIELDS:
        result.setdefault(field, None)

    if scenario == "full":
        return result
    if scenario == "description_only":
        for field in OPTIONAL_FIELDS:
            if field != "description":
                result[field] = None
        return result
    if scenario == "metadata_only":
        result["description"] = None
        return result
    if scenario == "no_component":
        result["component"] = None
        return result
    if scenario == "no_service":
        result["service"] = None
        return result
    if scenario == "no_request_type":
        result["request_type"] = None
        return result
    if scenario == "no_priority_criticality":
        result["priority"] = None
        result["criticality"] = None
        return result
    if scenario.startswith("random_dropout_"):
        probability = int(scenario.rsplit("_", 1)[1]) / 100.0
        return _drop_random_fields(result, probability=probability, seed=seed, scenario=scenario)
    if scenario == "empirical_missingness":
        rng = _stable_rng(seed, scenario, _row_key(result))
        for field, probability in EMPIRICAL_MISSINGNESS.items():
            if rng.random() < probability:
                result[field] = None
        return result

    raise AssertionError("unreachable")


def input_quality(values: Mapping[str, Any]) -> dict[str, Any]:
    available = []
    missing = []
    informative = 0
    for field in OPTIONAL_FIELDS:
        value = values.get(field)
        text = "" if value is None else str(value).strip()
        if text:
            available.append(field)
            informative += 1
        else:
            missing.append(field)

    completeness = informative / len(OPTIONAL_FIELDS)
    description = "" if values.get("description") is None else str(values.get("description")).strip()
    low_information = informative <= 2 or (not description and informative <= 3) or (description and len(description) < 8 and informative <= 3)
    return {
        "available_fields": available,
        "missing_fields": missing,
        "informative_field_count": informative,
        "input_completeness": round(completeness, 6),
        "low_information_warning": bool(low_information),
        "needs_more_information": bool(low_information),
    }
