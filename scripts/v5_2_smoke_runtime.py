from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "backend"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from app.ml.runtime import analyze_ticket, load_runtime, route_ticket  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke-test the integrated V5 category runtime")
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--profile", choices=("quality", "lite"), default="lite")
    args = parser.parse_args()

    runtime = load_runtime(args.models, model_profile=args.profile)
    payload = {
        "description": "Не получается войти в личный кабинет, ошибка авторизации после ввода кода.",
        "service": "Личный кабинет",
        "component": "Авторизация",
        "request_type": "Инцидент",
        "criticality": "Средняя",
        "urgency": "Средняя",
        "priority": "Средний",
    }
    full = analyze_ticket(runtime, payload, database=args.database)
    partial = analyze_ticket(
        runtime,
        {"description": "Не получается войти в личный кабинет, ошибка авторизации."},
        database=args.database,
    )
    confirmed_route = route_ticket(
        runtime,
        payload,
        confirmed_category=str(full["category"]["label"]),
        database=args.database,
    )

    category_pipeline = runtime["category"]["pipeline"]
    result = {
        "model_profile_requested": runtime.get("model_profile_requested"),
        "model_profile_active": runtime.get("model_profile_active"),
        "model_id": getattr(category_pipeline, "model_id", None),
        "model_revision": getattr(category_pipeline, "model_revision", None),
        "category": {
            "label": full["category"]["label"],
            "confidence": full["category"]["confidence"],
            "top3": full["category"]["alternatives"],
        },
        "routing": full["routing"],
        "confirmed_category_routing": confirmed_route,
        "partial_input": {
            "label": partial["category"]["label"],
            "confidence": partial["category"]["confidence"],
            "top3": partial["category"]["alternatives"],
            "routing": partial["routing"],
        },
        "similar_count": len(full.get("similar", [])),
        "model_provenance": full.get("model_provenance", {}),
    }
    if result["model_profile_active"] != args.profile:
        raise RuntimeError(f"Requested profile {args.profile!r} did not become active")
    if len(result["category"]["top3"]) != 3 or len(result["partial_input"]["top3"]) != 3:
        raise RuntimeError("Category TOP-3 contract failed")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
