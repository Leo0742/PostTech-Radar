from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "artifacts/gpu_research_v4/synthetic"
TOKEN_RE = re.compile(r"(?u)\b[\w-]{2,}\b")


def _hash(text: str) -> str:
    return hashlib.sha256(" ".join(text.lower().split()).encode("utf-8")).hexdigest()


def main() -> None:
    source = json.loads((ARTIFACTS / "synthetic_experiment.json").read_text(encoding="utf-8"))
    profiles = json.loads((ARTIFACTS / "real_train_category_profiles.json").read_text(encoding="utf-8"))["profiles"]
    by_class: dict[str, list[dict[str, Any]]] = defaultdict(list)
    rejected_reasons: Counter[str] = Counter()
    for sample in source["samples"]:
        label = str(sample["category"])
        text = str(sample["description"])
        profile = profiles[label]
        token_count = len(TOKEN_RE.findall(text))
        bounds = profile["style"]["tokens"]
        distinctive = {term.lower() for term in profile["distinctive_terms"]}
        normalized = text.lower()
        term_hits = sorted(term for term in distinctive if term in normalized)
        flags = []
        if token_count < max(4, bounds["p10"] // 2):
            flags.append("far_shorter_than_real_train")
        if bounds["p90"] and token_count > bounds["p90"] * 2:
            flags.append("far_longer_than_real_train")
        if distinctive and not term_hits:
            flags.append("no_distinctive_real_train_term")
        if label.lower() in normalized:
            flags.append("leaks_category_name")
        margin = float(sample.get("validation_filters", {}).get("embedding_margin", 0.0))
        if margin < 0.02:
            flags.append("low_semantic_margin")
        for flag in flags:
            rejected_reasons[flag] += 1
        record = {
            **sample,
            "sample_sha256": _hash(text),
            "audit_features": {
                "token_count": token_count,
                "real_train_token_range": bounds,
                "distinctive_term_hits": term_hits,
                "confusable_categories": profile["empirical_confusion_categories"],
                "automatic_flags": flags,
            },
            "manual_semantic_review": "PENDING",
            "manual_review_notes": "",
        }
        by_class[label].append(record)

    review_classes = []
    lines = [
        "# Synthetic-data semantic audit pack",
        "",
        "This pack is generated from frozen real TRAIN profiles. Automatic flags are triage only; acceptance requires semantic review.",
        "",
    ]
    for label in sorted(profiles):
        candidates = by_class.get(label, [])
        # Representative set: lowest margin, median margin, highest margin (when available).
        ordered = sorted(candidates, key=lambda item: float(item.get("validation_filters", {}).get("embedding_margin", 0.0)))
        selected = []
        if ordered:
            for index in sorted({0, len(ordered) // 2, len(ordered) - 1}):
                selected.append(ordered[index])
        profile = profiles[label]
        review_classes.append(
            {
                "category": label,
                "real_train_support": profile["real_train_support"],
                "ambiguity_risk": profile["ambiguity_risk"],
                "confusable_categories": profile["empirical_confusion_categories"],
                "generated_candidates": len(candidates),
                "representative_samples": selected,
                "class_decision": "PENDING",
                "class_review_notes": "",
            }
        )
        lines.extend(
            [
                f"## {label}",
                "",
                f"- Real TRAIN support: {profile['real_train_support']}",
                f"- Confusable: {', '.join(profile['empirical_confusion_categories']) or 'n/a'}",
                f"- Generated candidates after automatic filters: {len(candidates)}",
                "",
            ]
        )
        if not selected:
            lines.extend(["No accepted generator candidates to review.", ""])
        for position, sample in enumerate(selected, start=1):
            lines.extend(
                [
                    f"### Representative {position}",
                    "",
                    f"- SHA256: `{sample['sample_sha256']}`",
                    f"- Margin: {sample.get('validation_filters', {}).get('embedding_margin')}",
                    f"- Flags: {', '.join(sample['audit_features']['automatic_flags']) or 'none'}",
                    "",
                    str(sample["description"]),
                    "",
                ]
            )

    result = {
        "status": "PENDING_MANUAL_SEMANTIC_AUDIT",
        "source": "synthetic_experiment.json",
        "generator_model": source["generator_model"],
        "generator_revision": source["generator_revision"],
        "real_train_profiles": "real_train_category_profiles.json",
        "automatic_flags_summary": dict(rejected_reasons),
        "classes": review_classes,
    }
    (ARTIFACTS / "synthetic_audit_pack.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (ARTIFACTS / "synthetic_audit_pack.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"classes": len(review_classes), "samples": sum(len(v) for v in by_class.values())}, ensure_ascii=False))


if __name__ == "__main__":
    main()
