from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Sequence


def _load_synthetic(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        raw = json.loads(line)
        text = " ".join(str(raw.get("description") or "").split())
        label = " ".join(str(raw.get("target_category") or "").split())
        if not text or not label:
            continue
        rows.append(
            {
                "request_id": f"synthetic-v5-2-{raw.get('sample_sha256') or len(rows)}",
                "description": text,
                "category": label,
                "hard_negative_confusion_target": raw.get("hard_negative_confusion_target"),
                "source_weight": float(raw.get("synthetic_sample_weight") or 0.2),
                "quality_filters_passed": list(raw.get("quality_filters_passed") or []),
            }
        )
    return rows


def _select_synthetic(
    rows: Sequence[dict[str, Any]],
    train_support: Counter[str],
    labels: Sequence[str],
    *,
    allocation: str,
    ratio: float,
) -> list[dict[str, Any]]:
    by_label: dict[str, list[dict[str, Any]]] = defaultdict(list)
    label_set = set(labels)
    for row in rows:
        label = str(row["category"])
        if label in label_set and label != "Прочее":
            by_label[label].append(row)

    selected: list[dict[str, Any]] = []
    for label in labels:
        if label == "Прочее":
            continue
        support = int(train_support[str(label)])
        candidates = by_label.get(str(label), [])
        if not candidates:
            continue

        hard_pool = [row for row in candidates if row.get("hard_negative_confusion_target")]
        if allocation == "rare5":
            eligible = support <= 5
            pool = candidates
            cap = 12
        elif allocation == "rare20":
            eligible = support <= 20
            pool = candidates
            cap = 20
        elif allocation == "rare20_hard":
            eligible = support <= 20 or bool(hard_pool)
            pool = [*hard_pool, *[row for row in candidates if row not in hard_pool]]
            cap = 20
        else:
            raise ValueError(f"Unknown allocation: {allocation}")
        if not eligible:
            continue

        amount = min(len(pool), cap, max(1, int(math.ceil(max(1, support) * ratio))))
        selected.extend(pool[:amount])
    return selected
