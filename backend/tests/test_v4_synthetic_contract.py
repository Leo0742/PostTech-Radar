from __future__ import annotations

import importlib.util
import json
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]


def _protocol() -> dict:
    pointer_path = ROOT / "artifacts/gpu_research_v4/protocol.json"
    if not pointer_path.exists():
        pytest.skip("V4 research artifacts are not included in the lightweight source package")
    pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
    return json.loads((ROOT / pointer["protocol_path"]).read_text(encoding="utf-8"))


def test_every_taxonomy_label_has_a_structured_training_safe_definition() -> None:
    definitions = yaml.safe_load((ROOT / "data/category_definitions_v4.yaml").read_text(encoding="utf-8"))
    by_label = {item["category"]: item for item in definitions}
    assert set(by_label) == set(_protocol()["labels"])
    for item in definitions:
        assert item["meaning"]
        assert item["diagnostic_terms"]
        assert "does_not_belong" in item
        assert "confusable_categories" in item
        assert "key_difference_from_neighbors" in item
        assert "canonical_train_examples" in item
        assert item["definition_provenance"]["internet_text_used_as_training_data"] is False


def test_zero_real_train_definition_is_explicitly_official_and_training_safe() -> None:
    definitions = yaml.safe_load((ROOT / "data/category_definitions_v4.yaml").read_text(encoding="utf-8"))
    zero = next(item for item in definitions if item["category"] == "Тарификация посылок")
    assert zero["real_train_support"] == 0
    assert zero["canonical_train_examples"] == []
    assert zero["official_reference_context"]
    assert all(reference["url"].startswith("https://") for reference in zero["official_reference_context"])


def test_raw_synthetic_corpus_has_provenance_and_no_exact_duplicates() -> None:
    corpus = ROOT / "artifacts/gpu_research_v4/synthetic/domain_corpus_raw.jsonl"
    if not corpus.exists():
        pytest.skip("V4 synthetic corpus is not included in the lightweight source package")
    rows = [
        json.loads(line)
        for line in corpus.read_text(encoding="utf-8").splitlines()
        if line
    ]
    labels = set(_protocol()["labels"])
    hashes = [row["sample_sha256"] for row in rows]
    assert len(rows) >= 3_000
    assert len(hashes) == len(set(hashes))
    assert {row["target_category"] for row in rows} == labels
    for row in rows:
        assert row["generation_method"]
        assert row["prompt_template_version"]
        assert row["source_definition"] in labels
        assert row["quality_filters_passed"]
        assert row["synthetic_sample_weight"] <= 0.3


def test_deep_synthetic_rows_use_the_target_category_contract() -> None:
    script_path = ROOT / "scripts/v4/deep_synthetic_recheck.py"
    spec = importlib.util.spec_from_file_location("deep_synthetic_recheck", script_path)
    assert spec is not None and spec.loader is not None
    deep_synthetic_recheck = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(deep_synthetic_recheck)
    rows = deep_synthetic_recheck.build_synthetic_rows(
        source_samples=[
            {
                "target_category": "Тарификация посылок",
                "description": "не рассчитывается тариф посылки",
                "sample_sha256": "sample-1",
                "synthetic_sample_weight": 0.05,
            }
        ],
        approved_categories={"Тарификация посылок"},
        rejected_hashes=set(),
        prototypes={"Тарификация посылок": {"service": "Тарификатор"}},
    )

    assert rows == [
        {
            "service": "Тарификатор",
            "request_id": "synthetic-deep-0",
            "description": "не рассчитывается тариф посылки",
            "category": "Тарификация посылок",
            "is_synthetic": True,
            "synthetic_sample_weight": 0.05,
        }
    ]


def test_synthetic_finalist_ratio_selection_uses_normalized_category_rows() -> None:
    script_path = ROOT / "scripts/v4/deep_synthetic_finalists.py"
    spec = importlib.util.spec_from_file_location("deep_synthetic_finalists", script_path)
    assert spec is not None and spec.loader is not None
    deep_synthetic_finalists = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(deep_synthetic_finalists)
    samples = [
        {"category": "A", "description": "a1"},
        {"category": "A", "description": "a2"},
        {"category": "B", "description": "b1"},
    ]

    selected = deep_synthetic_finalists._choose(samples, Counter({"A": 1, "B": 1}), ratio=1)

    assert selected == [samples[0], samples[2]]


def test_zero_and_single_real_weight_caps_survive_grid_scaling() -> None:
    script_path = ROOT / "scripts/v4/deep_synthetic_finalists.py"
    spec = importlib.util.spec_from_file_location("deep_synthetic_finalists", script_path)
    assert spec is not None and spec.loader is not None
    deep_synthetic_finalists = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(deep_synthetic_finalists)

    assert deep_synthetic_finalists._effective_weight(
        {"synthetic_sample_weight": 0.05, "evidence_tier": "SYNTHETIC_ONLY"}, 2.0
    ) == 0.05
    assert deep_synthetic_finalists._effective_weight(
        {"synthetic_sample_weight": 0.10, "evidence_tier": "SINGLE_REAL_ANCHOR"}, 2.0
    ) == 0.10
    assert deep_synthetic_finalists._effective_weight(
        {"synthetic_sample_weight": 0.20, "evidence_tier": "REAL_ANCHORED"}, 2.0
    ) == 0.40


def test_final_retrain_preserves_zero_and_single_real_weight_caps() -> None:
    script_path = ROOT / "scripts/v4/final_evaluate_and_export.py"
    spec = importlib.util.spec_from_file_location("final_evaluate_and_export", script_path)
    assert spec is not None and spec.loader is not None
    final_evaluate_and_export = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(final_evaluate_and_export)

    assert final_evaluate_and_export._effective_synthetic_weight(
        {"synthetic_sample_weight": 0.05, "evidence_tier": "SYNTHETIC_ONLY"}, 2.0
    ) == 0.05
    assert final_evaluate_and_export._effective_synthetic_weight(
        {"synthetic_sample_weight": 0.10, "evidence_tier": "SINGLE_REAL_ANCHOR"}, 2.0
    ) == 0.10
    assert final_evaluate_and_export._effective_synthetic_weight(
        {"synthetic_sample_weight": 0.20, "evidence_tier": "REAL_ANCHORED"}, 2.0
    ) == 0.40


def test_clean_corpus_manifest_records_completed_manual_audit() -> None:
    manifest_path = ROOT / "artifacts/gpu_research_v4/synthetic/domain_corpus_clean_manifest.json"
    audit_path = ROOT / "artifacts/gpu_research_v4/synthetic/manual_synthetic_audit.json"
    if not manifest_path.exists() or not audit_path.exists():
        pytest.skip("V4 synthetic audit artifacts are not included in the lightweight source package")
    manifest = json.loads(
        manifest_path.read_text(encoding="utf-8")
    )
    audit = json.loads(audit_path.read_text(encoding="utf-8"))

    assert audit["status"] == "COMPLETE"
    assert manifest["status"] == "CLEAN_SEMANTIC_AUDIT_COMPLETE"
    assert manifest["manual_semantic_audit"] == "artifacts/gpu_research_v4/synthetic/manual_synthetic_audit.json"


def test_fastfit_receives_string_label_descriptions() -> None:
    script_path = ROOT / "scripts/v4/benchmark_fastfit.py"
    spec = importlib.util.spec_from_file_location("benchmark_fastfit", script_path)
    assert spec is not None and spec.loader is not None
    benchmark_fastfit = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(benchmark_fastfit)

    examples = benchmark_fastfit._fastfit_examples(
        [{"description": "не считается тариф", "category": "Тарификация посылок"}]
    )

    assert examples == {
        "text": ["не считается тариф"],
        "label": ["Тарификация посылок"],
    }


def test_fastfit_custom_dataset_supplies_compatibility_test_split() -> None:
    script_path = ROOT / "scripts/v4/benchmark_fastfit.py"
    spec = importlib.util.spec_from_file_location("benchmark_fastfit", script_path)
    assert spec is not None and spec.loader is not None
    benchmark_fastfit = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(benchmark_fastfit)
    train = [{"description": "train", "category": "A"}]
    validation = [{"description": "validation", "category": "A"}]

    splits = benchmark_fastfit._fastfit_splits(train, validation)

    assert splits["train"]["text"] == ["train"]
    assert splits["validation"]["text"] == ["validation"]
    assert splits["test"] == splits["validation"]


def test_fastfit_training_adapter_passes_ignore_keys_as_a_list() -> None:
    script_path = ROOT / "scripts/v4/benchmark_fastfit.py"
    spec = importlib.util.spec_from_file_location("benchmark_fastfit", script_path)
    assert spec is not None and spec.loader is not None
    benchmark_fastfit = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(benchmark_fastfit)

    captured: dict = {}

    class InnerTrainer:
        def train(self, **kwargs: object) -> str:
            captured.update(kwargs)
            return "trained"

    result = benchmark_fastfit._train_fastfit_compat(
        SimpleNamespace(trainer=InnerTrainer(), checkpoint=None)
    )

    assert result == "trained"
    assert captured["ignore_keys_for_eval"] == ["doc_input_ids", "doc_attention_mask", "labels"]


def test_neural_finalist_summary_reports_mean_and_stability() -> None:
    script_path = ROOT / "scripts/v4/summarize_neural_finalists.py"
    spec = importlib.util.spec_from_file_location("summarize_neural_finalists", script_path)
    assert spec is not None and spec.loader is not None
    summarize_neural_finalists = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(summarize_neural_finalists)
    runs = [
        {"accuracy": 0.7, "macro_f1": 0.5, "balanced_accuracy": 0.6, "worst_class_f1": 0.1},
        {"accuracy": 0.9, "macro_f1": 0.7, "balanced_accuracy": 0.8, "worst_class_f1": 0.3},
    ]

    summary = summarize_neural_finalists.summarize_metrics(runs)

    assert summary["accuracy"] == {"mean": 0.8, "std": 0.1}
    assert summary["macro_f1"] == {"mean": 0.6, "std": 0.1}


def test_neural_summary_excludes_unverified_lora_gradient_runs() -> None:
    script_path = ROOT / "scripts/v4/summarize_neural_finalists.py"
    spec = importlib.util.spec_from_file_location("summarize_neural_finalists", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert module._valid_training_run({"mode": "full"}) is True
    assert module._valid_training_run({"mode": "lora"}) is False
    assert module._valid_training_run(
        {"mode": "lora", "training_integrity": {"input_require_grads_enabled": True}}
    ) is True
