from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from scripts.v5_experiment import (
    EMBEDDING_DIMS,
    FEATURE_MODES,
    HEAD_REGISTRY,
    INSTRUCTION_REGISTRY,
    SEQUENCE_LENGTHS,
    build_metadata_prefix,
    build_stage_candidates,
    build_text,
    candidate_quality_key,
    deterministic_run_id,
    labels_for_view,
    load_json,
    operator_metrics,
    planned_folds,
    truncate_embeddings,
    validate_search_ids,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = PROJECT_ROOT / "artifacts" / "gpu_research_v5" / "protocol" / "protocol.json"
CONTRACT = PROJECT_ROOT / "artifacts" / "gpu_research_v5" / "data_contract.json"
SERVER_A_CONFIG = PROJECT_ROOT / "configs" / "v5" / "server_a_qwen8b.json"
SERVER_B_CONFIG = PROJECT_ROOT / "configs" / "v5" / "server_b_challengers.json"
REGISTRY_CONFIG = PROJECT_ROOT / "configs" / "v5" / "registry.json"


def test_frozen_gpu_search_registries_match_v51_spec() -> None:
    assert EMBEDDING_DIMS == (4096, 3072, 2048, 1024)
    assert SEQUENCE_LENGTHS == (128, 256, 384, 512, 768, 1024)
    assert set(INSTRUCTION_REGISTRY) == {
        "none",
        "ru_concise",
        "en_concise",
        "bilingual",
        "taxonomy_aware",
        "posttech_service_desk",
    }
    assert {
        "text_only",
        "metadata_prefix_text",
        "separate_metadata_text",
        "metadata_prior_text",
        "metadata_only_fallback",
        "text_only_fallback",
    }.issubset(FEATURE_MODES)
    assert {
        "logreg",
        "calibrated_logreg",
        "calibrated_linearsvc",
        "shallow_mlp",
        "centroid",
        "knn",
        "label_similarity",
        "metadata_fusion",
    }.issubset(HEAD_REGISTRY)


def test_category_views_are_exactly_top15_and_full43() -> None:
    contract = load_json(CONTRACT)

    top15 = labels_for_view(contract, "top15")
    full43 = labels_for_view(contract, "full43")

    assert len(top15) == 15
    assert len(full43) == 43
    assert set(top15).issubset(full43)


def test_lockbox_ids_are_rejected_from_search_partitions() -> None:
    protocol = load_json(PROTOCOL)
    lockbox_id = protocol["internal_lockbox_request_ids"][0]

    with pytest.raises(ValueError, match="lockbox"):
        validate_search_ids([lockbox_id], protocol)


def test_development_fold_plan_is_deterministic_and_lockbox_safe() -> None:
    protocol = load_json(PROTOCOL)

    first = planned_folds(protocol, repeats=[0], folds=[0, 1])
    second = planned_folds(protocol, repeats=[0], folds=[0, 1])

    assert first == second
    assert [(item["repeat"], item["fold"]) for item in first] == [(0, 0), (0, 1)]
    lockbox = set(protocol["internal_lockbox_request_ids"])
    assert not lockbox.intersection(first[0]["train_request_ids"])
    assert not lockbox.intersection(first[0]["validation_request_ids"])


def test_run_id_is_stable_and_sensitive_to_candidate_fields() -> None:
    base = {
        "stage": "smoke",
        "model_id": "Qwen/Qwen3-Embedding-8B",
        "view": "top15",
        "feature_mode": "text_only",
        "instruction": "none",
        "embedding_dim": 2048,
        "max_length": 256,
        "head": "logreg",
        "repeat": 0,
        "fold": 0,
    }

    assert deterministic_run_id(base) == deterministic_run_id(dict(base))
    changed = dict(base, max_length=384)
    assert deterministic_run_id(base) != deterministic_run_id(changed)


def test_text_rendering_preserves_missingness_and_never_prints_none() -> None:
    row = {
        "description": "Не работает QR код",
        "service": "ЛК",
        "component": None,
        "request_type": "Инцидент",
        "priority": None,
    }

    assert build_text(row, feature_mode="text_only") == "Не работает QR код"
    prefix = build_metadata_prefix(row)
    assert "None" not in prefix
    assert "component" not in prefix
    rendered = build_text(row, feature_mode="metadata_prefix_text")
    assert "Не работает QR код" in rendered
    assert "service: ЛК" in rendered


def test_embedding_dimension_truncation_is_explicit_and_validated() -> None:
    matrix = np.arange(24, dtype=float).reshape(3, 8)

    assert truncate_embeddings(matrix, 4).shape == (3, 4)
    with pytest.raises(ValueError, match="exceeds"):
        truncate_embeddings(matrix, 16)


def test_quality_key_follows_quality_first_selection_order() -> None:
    stronger_top1 = {
        "top1_accuracy": 0.80,
        "macro_f1": 0.70,
        "top3_accuracy": 0.90,
        "true_label_mrr": 0.85,
    }
    stronger_macro_only = {
        "top1_accuracy": 0.79,
        "macro_f1": 0.99,
        "top3_accuracy": 0.99,
        "true_label_mrr": 0.99,
    }

    assert candidate_quality_key(stronger_top1) > candidate_quality_key(stronger_macro_only)


def test_server_a_config_uses_staged_quality_first_funnel() -> None:
    config = load_json(SERVER_A_CONFIG)
    stages = {stage["name"]: stage for stage in config["stages"]}

    assert config["model"]["id"] == "Qwen/Qwen3-Embedding-8B"
    assert config["model"]["revision"] is None
    assert [stage["name"] for stage in config["stages"]] == [
        "smoke",
        "instruction_search",
        "representation_search",
        "feature_search",
        "head_tournament",
        "repeated_eval",
    ]
    assert stages["instruction_search"]["instructions"] == list(INSTRUCTION_REGISTRY)
    assert stages["representation_search"]["embedding_dims"] == list(EMBEDDING_DIMS)
    assert stages["representation_search"]["max_lengths"] == list(SEQUENCE_LENGTHS)
    assert stages["repeated_eval"]["repeats"] == [0, 1, 2]
    assert stages["repeated_eval"]["folds"] == [0, 1, 2, 3]


def test_server_b_config_has_fast_challenger_funnel() -> None:
    config = load_json(SERVER_B_CONFIG)

    assert config["funnel"] == ["smoke", "one_fold", "several_folds", "full_repeated"]
    assert {model["id"] for model in config["models"]} >= {
        "BAAI/bge-multilingual-gemma2",
        "nvidia/llama-embed-nemotron-8b",
    }
    assert any(model.get("deployment_policy") == "research_only_until_license_verified" for model in config["models"])


def test_stage_candidate_expansion_is_deterministic_and_uses_only_requested_fold() -> None:
    protocol = load_json(PROTOCOL)
    config = {
        "server": "A",
        "model": {"id": "Qwen/Qwen3-Embedding-8B", "revision": None},
        "stages": [
            {
                "name": "smoke",
                "views": ["top15"],
                "feature_modes": ["text_only"],
                "instructions": ["none", "ru_concise"],
                "embedding_dims": [2048],
                "max_lengths": [256],
                "heads": ["logreg"],
                "repeats": [0],
                "folds": [0],
            }
        ],
    }

    first = build_stage_candidates(config, "smoke", protocol=protocol)
    second = build_stage_candidates(config, "smoke", protocol=protocol)

    assert first == second
    assert len(first) == 2
    assert {(item["repeat"], item["fold"]) for item in first} == {(0, 0)}
    assert all(item["run_id"] for item in first)


def test_persisted_registry_matches_code_registry() -> None:
    registry = load_json(REGISTRY_CONFIG)

    assert registry["embedding_dims"] == list(EMBEDDING_DIMS)
    assert registry["sequence_lengths"] == list(SEQUENCE_LENGTHS)
    assert registry["instructions"] == INSTRUCTION_REGISTRY
    assert set(registry["feature_modes"]) == FEATURE_MODES
    assert set(registry["heads"]) == HEAD_REGISTRY


def test_operator_metrics_include_rank_distribution_and_topk() -> None:
    labels = ["A", "B", "C"]
    truth = ["A", "C", "B"]
    probabilities = np.asarray(
        [
            [0.8, 0.1, 0.1],
            [0.6, 0.1, 0.3],
            [0.2, 0.7, 0.1],
        ],
        dtype=float,
    )

    metrics = operator_metrics(truth, probabilities, labels)

    assert metrics["top1_accuracy"] == pytest.approx(2 / 3, abs=1e-6)
    assert metrics["top2_accuracy"] == 1.0
    assert metrics["top3_accuracy"] == 1.0
    assert metrics["rank_distribution"] == {"rank1": 2, "rank2": 1, "rank3": 0, "rank_gt3": 0}
    assert metrics["true_label_mrr"] == pytest.approx((1 + 0.5 + 1) / 3, abs=1e-6)
