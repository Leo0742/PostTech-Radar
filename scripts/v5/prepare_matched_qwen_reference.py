from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.v5_experiment import candidate_quality_key, load_json
from scripts.v5_gpu_runner import load_completed_stage_results, select_promoted_candidates


DEFAULT_RESULTS = ROOT / "artifacts" / "gpu_research_v5" / "runs" / "server_a"
DEFAULT_PROTOCOL = ROOT / "artifacts" / "gpu_research_v5" / "protocol" / "protocol.json"
DEFAULT_MANIFEST = ROOT / "artifacts" / "gpu_research_v5" / "matched_reference" / "frozen_recipe.json"
DEFAULT_4B_REVISION = "5cf2132abc99cad020ac570b19d031efec650f2b"
DEFAULT_4B_NATIVE_DIM = 2560
QUALITY_METRICS = ("top1_accuracy", "macro_f1", "top3_accuracy", "true_label_mrr")
RECIPE_FIELDS = ("instruction", "feature_mode", "embedding_dim", "max_length", "head")


def _recipe_key(candidate: dict[str, Any]) -> tuple[Any, ...]:
    return tuple(candidate[field] for field in RECIPE_FIELDS)


def _aggregate(records: list[dict[str, Any]]) -> dict[str, float]:
    return {
        metric: round(mean(float(record["metrics"][metric]) for record in records), 12)
        for metric in QUALITY_METRICS
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _config(
    *,
    server: str,
    artifact_root: str,
    model_id: str,
    model_revision: str,
    recipe: dict[str, Any],
    stage_name: str,
    views: list[str],
    repeats: list[int],
    folds: list[int],
) -> dict[str, Any]:
    return {
        "schema_version": "5.1",
        "server": server,
        "artifact_root": artifact_root,
        "model": {
            "id": model_id,
            "revision": model_revision,
            "deployment_policy": "matched_v5_reference",
        },
        "stages": [
            {
                "name": stage_name,
                "views": views,
                "feature_modes": [recipe["feature_mode"]],
                "instructions": [recipe["instruction"]],
                "embedding_dims": [int(recipe["embedding_dim"])],
                "max_lengths": [int(recipe["max_length"])],
                "heads": [recipe["head"]],
                "repeats": repeats,
                "folds": folds,
            }
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze final Qwen8B recipe and build a matched 4B/8B reference pair.")
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--qwen4b-revision", default=DEFAULT_4B_REVISION)
    parser.add_argument("--qwen4b-native-dim", type=int, default=DEFAULT_4B_NATIVE_DIM)
    args = parser.parse_args()

    head_results = load_completed_stage_results(args.results_root, "head_tournament")
    promoted = select_promoted_candidates(head_results, top_n=3)
    if len(promoted) != 3:
        raise RuntimeError(f"Expected three final promoted Qwen8B heads, got {len(promoted)}")

    promoted_keys = {_recipe_key(item["candidate"]): item for item in promoted}
    repeated = load_completed_stage_results(args.results_root, "repeated_eval")
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for record in repeated:
        candidate = record["candidate"]
        if candidate.get("view") != "top15":
            continue
        key = _recipe_key(candidate)
        if key in promoted_keys:
            grouped[key].append(record)

    candidates: list[dict[str, Any]] = []
    for key, promoted_item in promoted_keys.items():
        records = grouped.get(key, [])
        if len(records) != 12:
            raise RuntimeError(
                f"Final promoted recipe {key} has {len(records)} TOP15 repeated results; expected 12 before freezing."
            )
        aggregate = _aggregate(records)
        candidates.append(
            {
                "candidate": dict(promoted_item["candidate"]),
                "top15_repeated": aggregate,
                "quality_key": candidate_quality_key(aggregate),
            }
        )
    candidates.sort(key=lambda item: item["quality_key"], reverse=True)
    winner = candidates[0]
    winner_candidate = winner["candidate"]
    winner_key = _recipe_key(winner_candidate)
    synthetic_recipe = winner_candidate.get("synthetic_recipe")
    if synthetic_recipe not in (None, "none", {}, []):
        raise RuntimeError(
            "The frozen Qwen8B winner contains a synthetic recipe, but the current V5 GPU runner "
            "does not apply synthetic training rows. Refusing to create an unmatched Qwen4B reference."
        )

    full43 = [
        record
        for record in repeated
        if record["candidate"].get("view") == "full43" and _recipe_key(record["candidate"]) == winner_key
    ]
    if len(full43) != 12:
        raise RuntimeError(f"Winning recipe has {len(full43)} FULL43 repeated results; expected 12 before freezing.")

    protocol = load_json(args.protocol)
    final_dim = int(winner_candidate["embedding_dim"])
    common_dim = min(final_dim, int(args.qwen4b_native_dim))
    recipe = {
        "instruction": winner_candidate["instruction"],
        "feature_mode": winner_candidate["feature_mode"],
        "embedding_dim": common_dim,
        "winning_qwen8b_embedding_dim": final_dim,
        "embedding_dim_adjusted_for_matching": common_dim != final_dim,
        "max_length": int(winner_candidate["max_length"]),
        "head": winner_candidate["head"],
        "calibration": (
            winner_candidate["head"]
            if str(winner_candidate["head"]).startswith("calibrated_")
            else "none_beyond_head_implementation"
        ),
        "missing_data_handling": "__MISSING__ structured category",
        "structured_metadata_branch": winner_candidate["feature_mode"],
        "synthetic_recipe": synthetic_recipe,
        "synthetic_policy": "no synthetic training rows used by the frozen V5 GPU recipe",
    }

    qwen8b_revision = str(winner_candidate["model_revision"])
    configs = {
        "matched_qwen8b": _config(
            server="MATCHED_QWEN8B",
            artifact_root="artifacts/gpu_research_v5/runs/matched_qwen8b",
            model_id="Qwen/Qwen3-Embedding-8B",
            model_revision=qwen8b_revision,
            recipe=recipe,
            stage_name="matched_eval",
            views=["top15", "full43"],
            repeats=[0, 1, 2],
            folds=[0, 1, 2, 3],
        ),
        "matched_qwen4b": _config(
            server="MATCHED_QWEN4B",
            artifact_root="artifacts/gpu_research_v5/runs/matched_qwen4b",
            model_id="Qwen/Qwen3-Embedding-4B",
            model_revision=str(args.qwen4b_revision),
            recipe=recipe,
            stage_name="matched_eval",
            views=["top15", "full43"],
            repeats=[0, 1, 2],
            folds=[0, 1, 2, 3],
        ),
        "benchmark_qwen8b": _config(
            server="MATCHED_QWEN8B_BENCH",
            artifact_root="artifacts/gpu_research_v5/runs/matched_benchmark_qwen8b",
            model_id="Qwen/Qwen3-Embedding-8B",
            model_revision=qwen8b_revision,
            recipe=recipe,
            stage_name="matched_benchmark",
            views=["top15"],
            repeats=[0],
            folds=[0],
        ),
        "benchmark_qwen4b": _config(
            server="MATCHED_QWEN4B_BENCH",
            artifact_root="artifacts/gpu_research_v5/runs/matched_benchmark_qwen4b",
            model_id="Qwen/Qwen3-Embedding-4B",
            model_revision=str(args.qwen4b_revision),
            recipe=recipe,
            stage_name="matched_benchmark",
            views=["top15"],
            repeats=[0],
            folds=[0],
        ),
    }

    config_paths = {
        "matched_qwen8b": ROOT / "configs" / "v5" / "matched_qwen8b.json",
        "matched_qwen4b": ROOT / "configs" / "v5" / "matched_qwen4b.json",
        "benchmark_qwen8b": ROOT / "configs" / "v5" / "matched_benchmark_qwen8b.json",
        "benchmark_qwen4b": ROOT / "configs" / "v5" / "matched_benchmark_qwen4b.json",
    }
    for key, path in config_paths.items():
        _write_json(path, configs[key])

    manifest = {
        "status": "frozen",
        "purpose": "isolate Qwen3-Embedding-8B model-size gain from V5 pipeline gain",
        "selection_rule": "final head-tournament top 3, then repeated TOP15 quality key (TOP1, macro-F1, TOP3, MRR)",
        "winning_qwen8b_recipe": {field: winner_candidate[field] for field in RECIPE_FIELDS},
        "winning_qwen8b_revision": qwen8b_revision,
        "winning_qwen8b_top15_repeated": winner["top15_repeated"],
        "winning_qwen8b_full43_repeated": _aggregate(full43),
        "matched_recipe": recipe,
        "qwen4b_revision": str(args.qwen4b_revision),
        "qwen4b_native_embedding_dim": int(args.qwen4b_native_dim),
        "common_embedding_dim": common_dim,
        "protocol_version": protocol.get("protocol_version"),
        "dataset_sha256": protocol.get("dataset_sha256"),
        "split_sha256": protocol.get("split_sha256"),
        "repeats": [0, 1, 2],
        "folds": [0, 1, 2, 3],
        "views": ["top15", "full43"],
        "configs": {key: str(path.relative_to(ROOT)) for key, path in config_paths.items()},
    }
    _write_json(args.manifest, manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
