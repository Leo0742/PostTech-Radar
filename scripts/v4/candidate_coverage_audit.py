from __future__ import annotations

import json
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_ROOT = PROJECT_ROOT / "artifacts/gpu_research_v4"


def _read(relative: str) -> dict[str, Any]:
    path = ARTIFACT_ROOT / relative
    if not path.exists():
        raise FileNotFoundError(f"required measured artifact is missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload


def _summary(relative: str) -> dict[str, Any]:
    payload = _read(relative)
    metrics = payload.get("metrics", {})
    metric_variant = "direct"
    if "macro_f1" not in metrics and "embedding_metadata_lr" in metrics:
        metric_variant = "embedding_metadata_lr"
        metrics = metrics[metric_variant]
    elif "macro_f1" not in metrics and ("text_metadata_80_20" in metrics or "text" in metrics):
        metric_variant = "text_metadata_80_20" if "text_metadata_80_20" in metrics else "text"
        metrics = metrics[metric_variant]
    elif "macro_f1" not in metrics and payload.get("selected", {}).get("metrics"):
        metric_variant = "selected"
        metrics = payload["selected"]["metrics"]
    if "macro_f1" not in metrics:
        raise ValueError(f"artifact has no comparable classification metrics: {relative}")
    return {
        "path": f"artifacts/gpu_research_v4/{relative}",
        "candidate_id": payload.get("candidate_id"),
        "model_id": payload.get("model_id"),
        "view": payload.get("view"),
        "status": payload.get("status"),
        "metric_variant": metric_variant,
        "accuracy": metrics.get("accuracy"),
        "macro_f1": metrics.get("macro_f1"),
        "balanced_accuracy": metrics.get("balanced_accuracy"),
        "worst_class_f1": metrics.get("worst_class_f1"),
    }


def _optional_summaries(*relatives: str) -> list[dict[str, Any]]:
    return [_summary(relative) for relative in relatives if (ARTIFACT_ROOT / relative).exists()]


def main() -> None:
    pointer = json.loads((ARTIFACT_ROOT / "protocol.json").read_text(encoding="utf-8"))
    protocol = json.loads((PROJECT_ROOT / pointer["protocol_path"]).read_text(encoding="utf-8"))
    ruroberta_artifacts = [
        _summary("coverage/ruroberta-large-top15-f0-s20260917-research-license-unresolved.json"),
        *_optional_summaries(
            "finetune/ruroberta-large-lora-top15-f1-s20260918-deep-fold1-balanced.json",
            "finetune/ruroberta-large-lora-top15-f2-s20260919-deep-fold2-focal.json",
            "finetune/ruroberta-large-lora-full43-f0-s20260917-deep-full43.json",
        ),
    ]
    ruroberta_promoted = len(ruroberta_artifacts) > 1
    autointent_artifacts = [
        _summary("coverage/autointent-classic-light-top15-f0-s20260917.json"),
        *_optional_summaries(
            "coverage/autointent-classic-light-top15-f1-s20260918.json",
            "coverage/autointent-classic-light-top15-f2-s20260919.json",
        ),
    ]
    autointent_promoted = len(autointent_artifacts) > 1
    candidates = [
        {
            "requested_family": "FacebookAI/xlm-roberta-large",
            "coverage": "MEASURED",
            "role": "plain classifier; full fine-tuning and corrected LoRA/deep variants",
            "artifacts": [
                _summary("finetune/xlm-roberta-large-deep-full-top15-f0-s20260917-full-lr1e5.json"),
                _summary("finetune/xlm-roberta-large-deep-lora-full43-f0-s20260917-full43-f0-focal.json"),
            ],
            "rerun": False,
        },
        {
            "requested_family": "ai-forever/ruBert-base",
            "coverage": "MEASURED",
            "role": "real measured full fine-tuning baseline",
            "artifacts": [
                _summary("finetune/rubert-base-full-top15-f0-s20260917.json"),
                _summary("finetune/rubert-base-full-full43-f0-s20260917.json"),
            ],
            "rerun": False,
        },
        {
            "requested_family": "ai-forever/ruRoberta-large",
            "coverage": "MEASURED_RESEARCH_ONLY_DEEP" if ruroberta_promoted else "MEASURED_RESEARCH_ONLY",
            "role": "frozen-fold screening; excluded from deployment while license is unresolved",
            "artifacts": ruroberta_artifacts,
            "rerun": True,
            "deployment_eligibility": "BLOCKED_LICENSE_UNRESOLVED",
            "promotion_threshold_macro_f1": 0.65,
            "promotion_decision": "PROMOTED_TO_DEEP" if ruroberta_promoted else "STOPPED_AFTER_SCREENING_NONCOMPETITIVE",
        },
        {
            "requested_family": "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli",
            "coverage": "MEASURED",
            "role": "label-description / NLI entailment scoring (not plain softmax)",
            "artifacts": [_summary("nli/mdeberta-top15.json"), _summary("nli/mdeberta-full43.json")],
            "rerun": False,
        },
        {
            "requested_family": "Qwen3-Embedding-0.6B / 4B",
            "coverage": "MEASURED_BOTH_SIZES",
            "role": "embedding classifier and deep head",
            "artifacts": [
                _summary("embeddings/qwen3-embedding-06b-top15.json"),
                _summary("embeddings/qwen3-embedding-06b-full43.json"),
                _summary("deep/qwen3-embedding-4b-top15.json"),
                _summary("deep/qwen3-embedding-4b-full43.json"),
            ],
            "rerun": False,
        },
        {
            "requested_family": "BAAI/bge-m3",
            "coverage": "MEASURED",
            "role": "embedding classifier and deep head",
            "artifacts": [_summary("deep/bge-m3-top15.json"), _summary("deep/bge-m3-full43.json")],
            "rerun": False,
        },
        {
            "requested_family": "intfloat/multilingual-e5-large-instruct",
            "coverage": "MEASURED",
            "role": "embedding classifier and retrieval",
            "artifacts": [
                _summary("embeddings/e5-large-instruct-top15.json"),
                _summary("embeddings/e5-large-instruct-full43.json"),
            ],
            "rerun": False,
        },
        {
            "requested_family": "FastFit",
            "coverage": "MEASURED",
            "role": "supervised contrastive / few-shot classifier",
            "artifacts": [
                _summary("fastfit/xlmr-base-deep-top15-f0-s20260917-base.json"),
                _summary("fastfit/xlmr-base-deep-full43-f1-s20260918-full43.json"),
            ],
            "rerun": False,
        },
        {
            "requested_family": "DeepPavlov AutoIntent",
            "coverage": "MEASURED_DEEP" if autointent_promoted else "MEASURED",
            "role": "AutoML intent classification with classic-light preset on frozen fold",
            "artifacts": autointent_artifacts,
            "rerun": True,
            "promotion_threshold_macro_f1": 0.65,
            "promotion_decision": "PROMOTED_TO_REPEATED_FOLDS" if autointent_promoted else "STOPPED_AFTER_SCREENING_NONCOMPETITIVE",
            "full43_note": "AutoIntent 0.3.2 requires every class in every internal validation split; frozen FULL43 folds lack some rare classes, so no contaminated/dummy validation rows were introduced.",
        },
    ]
    payload = {
        "audit": "candidate-coverage-v4",
        "status": "COMPLETE",
        "dataset_sha256": protocol["dataset_sha256"],
        "split_sha256": protocol["split_sha256"],
        "sealed_holdout_accessed": False,
        "selection_and_holdout_blocked_until_complete": False,
        "planned_family_count": len(candidates),
        "measured_family_count": len(candidates),
        "candidates": candidates,
        "conclusion": (
            "All nine planned families have measured artifacts. Existing comparable runs were not repeated. "
            "ruRoberta-large remains research-only because its deployment license is unresolved."
        ),
    }
    output = ARTIFACT_ROOT / "final/candidate-coverage-audit.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# Candidate coverage audit v4",
        "",
        f"Frozen dataset SHA-256: `{payload['dataset_sha256']}`  ",
        f"Frozen split SHA-256: `{payload['split_sha256']}`  ",
        "Sealed holdout accessed: **no**",
        "",
        "| Planned family | Coverage | Intended/measured role | Evidence |",
        "|---|---:|---|---|",
    ]
    for item in candidates:
        evidence = ", ".join(
            f"`{artifact['path']}` (macro-F1 {artifact['macro_f1']})" for artifact in item["artifacts"]
        )
        lines.append(
            f"| {item['requested_family']} | {item['coverage']} | {item['role']} | {evidence} |"
        )
    lines.extend(
        [
            "",
            "No valid comparable artifact was rerun. The two accidental omissions were screened before finalist "
            "selection. ruRoberta-large is evidence-only and cannot be selected for deployment until its license "
            "is resolved.",
            "",
        ]
    )
    report = PROJECT_ROOT / "outputs/CANDIDATE_COVERAGE_AUDIT_V4.md"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"artifact": str(output), "report": str(report), "families": len(candidates)}))


if __name__ == "__main__":
    main()
