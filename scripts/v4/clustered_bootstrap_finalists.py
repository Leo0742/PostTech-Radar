from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, f1_score

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

from app.core.config import DATABASE_PATH  # noqa: E402
from app.services.data_service import load_rows  # noqa: E402


def _group(row: dict[str, object]) -> str:
    value = " ".join(str(row.get("normalized_description") or row.get("description") or "").lower().split())
    return hashlib.sha256(value.encode()).hexdigest()[:20]


def main() -> None:
    stack_path = PROJECT_ROOT / "artifacts/gpu_research_v4/deep/stack-top15.json"
    stack_payload = json.loads(stack_path.read_text(encoding="utf-8"))
    champion = {
        (int(row["repeat"]), str(row["request_id"])): str(row["prediction"])
        for row in stack_payload["predictions"]
    }
    truth = {
        (int(row["repeat"]), str(row["request_id"])): str(row["truth"])
        for row in stack_payload["predictions"]
    }
    rows = {str(row["request_id"]): row for row in load_rows(DATABASE_PATH)}
    groups = {request_id: _group(rows[request_id]) for _, request_id in champion}
    comparisons = {}
    rng = np.random.default_rng(20260917)
    for path in sorted((PROJECT_ROOT / "artifacts/gpu_research_v4/deep").glob("*-top15.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("status") != "MEASURED_DEEP_OPTIMIZATION" or not payload.get("predictions"):
            continue
        challenger = {
            (int(row["repeat"]), str(row["request_id"])): str(row["prediction"])
            for row in payload["predictions"]
        }
        common = sorted(set(champion) & set(challenger))
        group_names = sorted({groups[key[1]] for key in common})
        keys_by_group = {
            group: [key for key in common if groups[key[1]] == group] for group in group_names
        }
        samples = []
        for _ in range(1000):
            selected = rng.choice(group_names, size=len(group_names), replace=True)
            sampled_keys = [key for group in selected for key in keys_by_group[str(group)]]
            targets = [truth[key] for key in sampled_keys]
            champion_values = [champion[key] for key in sampled_keys]
            challenger_values = [challenger[key] for key in sampled_keys]
            samples.append(
                (
                    accuracy_score(targets, champion_values) - accuracy_score(targets, challenger_values),
                    f1_score(targets, champion_values, average="macro", zero_division=0)
                    - f1_score(targets, challenger_values, average="macro", zero_division=0),
                )
            )
        values = np.asarray(samples)
        comparisons[payload["candidate_id"]] = {
            "clusters": len(group_names),
            "observations_including_repeats": len(common),
            "accuracy_delta_mean": round(float(values[:, 0].mean()), 6),
            "accuracy_delta_ci95": [round(float(value), 6) for value in np.quantile(values[:, 0], [0.025, 0.975])],
            "macro_f1_delta_mean": round(float(values[:, 1].mean()), 6),
            "macro_f1_delta_ci95": [round(float(value), 6) for value in np.quantile(values[:, 1], [0.025, 0.975])],
            "probability_champion_better_accuracy": round(float((values[:, 0] > 0).mean()), 6),
            "probability_champion_better_macro_f1": round(float((values[:, 1] > 0).mean()), 6),
        }
    result = {
        "status": "MEASURED_CLUSTERED_BOOTSTRAP",
        "champion_candidate": stack_payload["candidate_id"],
        "bootstrap_iterations": 1000,
        "cluster": "normalized_description_group; repeated-CV observations stay together",
        "comparisons": comparisons,
    }
    destination = PROJECT_ROOT / "artifacts/gpu_research_v4/deep/clustered-bootstrap-top15.json"
    destination.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
