from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

from app.core.config import DATABASE_PATH  # noqa: E402
from app.ml.v3.candidates import CandidateSpec  # noqa: E402
from app.ml.v3.dataset import challenge_dataset_config, training_corpus_summary  # noqa: E402
from app.ml.v3.experiments import evaluate_candidate  # noqa: E402
from app.ml.v3.protocol import audit_manifest, build_v3_manifest, manifest_from_dict  # noqa: E402
from app.ml.v3.registry import CandidateRegistry  # noqa: E402
from app.services.data_service import load_rows  # noqa: E402

CANDIDATES = {
    "tfidf_lr": ("tfidf_lr", "text", {}),
    "metadata_lr": ("structured_lr", "metadata", {}),
    "structured_lr": ("structured_lr", "combined", {}),
    "calibrated_linear_svc": ("calibrated_linear_svc", "combined", {}),
    "catboost_text": ("catboost_text", "combined", {"iterations": 80, "depth": 6, "learning_rate": 0.05}),
}


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _task_rows(database: Path, task: str) -> list[dict[str, object]]:
    rows = load_rows(database)
    if task == "category":
        summary = training_corpus_summary(database, challenge_dataset_config())
        labels = {item["name"] for item in summary["top_k"]}
        return [row for row in rows if row["category"] in labels and row["description"].strip()]
    allowed = {"(1 линия)", "(2 линия)", "(3 линия)"}
    return [{**row, "category": row["final_line"]} for row in rows if row["final_line"] in allowed and row["description"].strip()]


def _load_or_create_protocol(path: Path, rows: list[dict[str, object]], seed: int) -> object:
    if path.exists():
        return manifest_from_dict(json.loads(path.read_text(encoding="utf-8")))
    manifest = build_v3_manifest(rows, seed=seed)
    _write_json(path, manifest.to_dict() | {"audit": audit_manifest(manifest), "sealed": True})
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a development-only PostTech Radar v3 classical candidate")
    parser.add_argument("candidate", choices=(*CANDIDATES, "all"))
    parser.add_argument("--task", choices=("category", "routing"), default="category")
    parser.add_argument("--database", type=Path, default=DATABASE_PATH)
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "artifacts/research/v3")
    args = parser.parse_args()
    rows = _task_rows(args.database, args.task)
    protocol_name = "protocol.json" if args.task == "category" else "protocol_routing.json"
    protocol_path = PROJECT_ROOT / "artifacts/evaluation/v3" / protocol_name
    manifest = _load_or_create_protocol(protocol_path, rows, 20260916 if args.task == "category" else 20261007)
    registry = CandidateRegistry(args.output_root / "registry.json")
    names = list(CANDIDATES) if args.candidate == "all" else [args.candidate]
    completed = []
    for name in names:
        family, feature_mode, parameters = CANDIDATES[name]
        candidate_id = f"{args.task}/{name}/v3"
        result_path = args.output_root / "candidates" / f"{args.task}-{name}.json"
        if result_path.exists():
            result = json.loads(result_path.read_text(encoding="utf-8"))
        else:
            spec = CandidateSpec(candidate_id, args.task, family, feature_mode, "A", parameters)
            target = "category" if args.task == "category" else "final_line"
            result = evaluate_candidate(spec, rows, manifest, target=target)
            _write_json(result_path, result)
        registry_record = {key: result[key] for key in (
            "candidate_id", "status", "tier", "task", "family", "feature_mode", "parameters",
            "dataset_sha256", "split_sha256", "development_only", "sealed_holdout_accessed", "package_versions", "metrics",
        )}
        registry_record["result_path"] = str(result_path.relative_to(PROJECT_ROOT))
        registry.append(registry_record)
        completed.append({"candidate_id": candidate_id, "macro_f1": result["metrics"]["cv_macro_f1_mean"]})
    print(json.dumps({"completed": completed, "protocol": str(protocol_path)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
