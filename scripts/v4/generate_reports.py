from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
ART = ROOT / "artifacts/gpu_research_v4"
OUT = ROOT / "outputs"


def load(path: Path) -> dict[str, Any] | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def write(name: str, text: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(text.strip() + "\n", encoding="utf-8")


def pct(value: Any) -> str:
    return "—" if value is None else f"{100 * float(value):.2f}%"


def metric_row(name: str, view: str, metrics: dict[str, Any], note: str = "") -> str:
    return (
        f"| {name} | {view} | {pct(metrics.get('accuracy'))} | {pct(metrics.get('macro_f1'))} | "
        f"{pct(metrics.get('balanced_accuracy'))} | {pct(metrics.get('worst_class_f1'))} | {note} |"
    )


def comparison_rows() -> list[str]:
    rows = []
    for path in sorted((ART / "embeddings").glob("*.json")):
        item = load(path)
        if not item:
            continue
        metrics = item.get("metrics", {}).get("embedding_metadata_lr")
        if metrics:
            note = f"screening; {item.get('encode_ms_per_ticket', '—')} ms/ticket; VRAM {item.get('peak_vram_bytes', 0)/2**30:.2f} GiB"
            rows.append(metric_row(str(item["model_id"]), str(item["view"]), metrics, note))
    for path in sorted((ART / "deep").glob("*.json")):
        item = load(path)
        if not item or item.get("status") not in {"MEASURED_DEEP_OPTIMIZATION", "MEASURED_DEEP_STRICT_OOF"}:
            continue
        stability = item.get("stability")
        if stability:
            metrics = {key: value["mean"] for key, value in stability.items()}
            note = "deep; mean over 3 repeated grouped-CV runs"
            rows.append(metric_row(str(item["candidate_id"]), str(item.get("view", "")), metrics, note))
    return rows


def main() -> None:
    environment = load(ART / "server_environment.json") or {}
    protocol_pointer = load(ART / "protocol.json") or {}
    protocol = load(ROOT / str(protocol_pointer.get("protocol_path", ""))) or {}
    ood = load(ART / "ood/leave-category-out.json") or {}
    routing = load(ART / "routing/oof-category.json") or {}
    synthetic = load(ART / "deep/synthetic-recheck-full43.json") or load(
        ART / "synthetic/synthetic_experiment.json"
    ) or {}
    synthetic_raw = load(ART / "synthetic/domain_corpus_raw_manifest.json") or {}
    synthetic_clean = load(ART / "synthetic/domain_corpus_clean_manifest.json") or {}
    synthetic_finalists = load(ART / "deep/synthetic-finalists.json") or {}
    synthetic_audit = load(ART / "synthetic/manual_synthetic_audit.json") or {}
    synthetic_family_summary = {
        name: {key: values.get(key) for key in ("accepted", "baseline", "best")}
        for name, values in synthetic_finalists.get("families", {}).items()
    }
    specialist = load(ART / "specialist/qwen3-8b-top15.json") or {}
    neural_finalists = load(ART / "deep/neural-finalist-summary.json") or {}
    coverage_audit = load(ART / "final/candidate-coverage-audit.json") or {}
    final = load(ART / "final/final-evaluation.json") or {}
    registry = load(ROOT / "models/v4/registry.json") or {}
    active = registry.get("ACTIVE_CHAMPION", {})

    experiments = comparison_rows()
    for candidate_id, item in neural_finalists.get("candidates", {}).items():
        metrics = {key: value["mean"] for key, value in item["stability"].items()}
        experiments.append(
            metric_row(
                candidate_id,
                str(item["best_run"]["view"]),
                metrics,
                f"deep neural; {item['run_count']} frozen fold/seed runs; not full 12-fold",
            )
        )
    nli = load(ART / "nli/mdeberta-full43.json")
    if nli:
        experiments.append(metric_row(str(nli["model_id"]), "full43", nli["metrics"], "pure label-semantic screening"))
    for path in sorted((ART / "fastfit").glob("*.json")):
        item = load(path)
        if item:
            experiments.append(metric_row(f"FastFit / {item['model_id']}", str(item["view"]), item["metrics"], "frozen fold"))

    write(
        "GPU_RESEARCH_V4.md",
        f"""
# GPU Research V4 — ПочтаТех Радар

## Server

- Host: `{environment.get('hostname', '—')}`; {environment.get('os', '—')}.
- GPU: {environment.get('gpu', '—')}, {environment.get('vram_gib', '—')} GiB; driver {environment.get('nvidia_driver', '—')}; CUDA runtime {environment.get('cuda_driver_runtime', '—')}.
- Python {environment.get('python_system', '—')}; PyTorch {environment.get('pytorch', '—')}.
- CPU/RAM: {environment.get('cpu', '—')}; {environment.get('ram_gib', '—')} GiB.

## Honest protocol

- Dataset SHA-256: `{protocol.get('dataset_sha256', '—')}`.
- Split SHA-256: `{protocol.get('split_sha256', '—')}`.
- Real rows: {protocol.get('real_row_count', '—')}; labels: {len(protocol.get('labels', []))}.
- Repeated StratifiedGroupKFold: {len(protocol.get('folds', []))} frozen folds.
- Calibration, grouped holdout, temporal evaluation, retrieval development/test and leave-category-out OOD are separate.
- Synthetic samples are excluded from every validation, calibration, temporal and final-test partition.
- Candidate coverage before finalist selection: {coverage_audit.get('measured_family_count', '—')}/{coverage_audit.get('planned_family_count', '—')} planned families; status `{coverage_audit.get('status', '—')}`.
- The sealed holdout remained closed during coverage audit: `{not bool(coverage_audit.get('sealed_holdout_accessed', True))}`.

## Experiments actually measured

| Candidate | View | Accuracy | Macro-F1 | Balanced accuracy | Worst F1 | Note |
|---|---:|---:|---:|---:|---:|---|
{chr(10).join(experiments)}

## Final result

- Active production champion: `{active.get('candidate_id', final.get('candidate_id', 'not frozen'))}`.
- Final sealed evaluation: {json.dumps(final.get('metrics', {}), ensure_ascii=False)}.
- Runtime evidence: {json.dumps(final.get('runtime_verification', {}), ensure_ascii=False)}.

## Limitations

- Classes represented by one to five independent real groups cannot yield a stable supervised estimate; their metrics are reported, not hidden.
- Human retrieval labels were not available. The generated review CSV intentionally leaves labels blank.
- External Yandex/GigaChat APIs were not tested because credentials were not supplied; local research did not wait for them.
- Pure NLI and shallow XLM-R screening were weak; deep results are kept separate from screening.
- `ai-forever/ruRoberta-large` is research-only evidence and is deployment-ineligible until its license is explicitly resolved.
- Full per-family evidence and reasons are recorded in `outputs/CANDIDATE_COVERAGE_AUDIT_V4.md` and `artifacts/gpu_research_v4/final/candidate-coverage-audit.json`.
""",
    )

    write(
        "MODEL_COMPARISON_V4.md",
        f"""
# Model Comparison V4

The table distinguishes broad screening from deep optimization. Hyperparameters were selected only on frozen development data; repeated confirmation and temporal checks precede the one-time sealed evaluation.

| Candidate | View | Accuracy | Macro-F1 | Balanced accuracy | Worst F1 | Note |
|---|---:|---:|---:|---:|---:|---|
{chr(10).join(experiments)}

## Selection rule

TOP15 uses accuracy, macro-F1 and worst-class F1. FULL43 adds balanced accuracy and support-band F1. The decision also includes temporal robustness, clustered bootstrap, latency, VRAM and deployability. A heavy stack is not promoted for a statistically unclear marginal gain.

## Registry

```json
{json.dumps(registry, ensure_ascii=False, indent=2)}
```
""",
    )

    full43_source = (
        load(ART / "deep/stack-full43.json")
        or load(ART / "deep/structured-full43.json")
        or load(ART / "structured/full43.json")
        or {}
    )
    full43_metrics = full43_source.get("repeat_metrics", {}).get("0") or full43_source.get("metrics", {})
    per_class_lines = [
        f"| {label} | {values['support']} | {pct(values['precision'])} | {pct(values['recall'])} | {pct(values['f1'])} |"
        for label, values in full43_metrics.get("per_class", {}).items()
    ]
    write(
        "FULL43_RESEARCH_V4.md",
        f"""
# FULL43 and rare-class research

Source candidate: `{full43_source.get('candidate_id', '—')}`.

- Accuracy: {pct(full43_metrics.get('accuracy'))}
- Macro-F1: {pct(full43_metrics.get('macro_f1'))}
- Balanced accuracy: {pct(full43_metrics.get('balanced_accuracy'))}
- Worst-class F1: {pct(full43_metrics.get('worst_class_f1'))}
- Support bands: `{json.dumps(full43_metrics.get('support_bands', {}), ensure_ascii=False)}`

Category definitions, real-example centroids, kNN, FastFit, mDeBERTa NLI, deep embedding heads, rare-semantic stacking and controlled synthetic augmentation were measured. Zero F1 for singleton categories is retained because synthetic text cannot create independent real evidence.

Training-evidence bands (0 real / 1 real / 2–5 real) are reported separately in:

```json
{json.dumps(synthetic_finalists.get('training_evidence_bands', {}), ensure_ascii=False, indent=2)}
```

| Category | Real support | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
{chr(10).join(per_class_lines)}
""",
    )

    write(
        "SYNTHETIC_DATA_V4.md",
        f"""
# Synthetic Data V4

## Corpus

- Sources: controlled domain-ontology/slot generation plus the already-running local `Qwen/Qwen3-8B` generation; no hosted inference API.
- Raw controlled corpus: {synthetic_raw.get('rows', '—')} rows; {synthetic_raw.get('hard_confusion_rows', '—')} hard-confusion rows.
- Clean corpus after semantic/style/contamination/diversity QC: {synthetic_clean.get('rows_after_qc', '—')} of {synthetic_clean.get('rows_before_qc', '—')} rows.
- Clean manifest: `{json.dumps(synthetic_clean, ensure_ascii=False)}`.
- Representative semantic audit: `{json.dumps(synthetic_audit.get('summary', {}), ensure_ascii=False)}`.

Every retained row stores target class, generation method/model, prompt or template, revision where applicable, seed, definition snapshot, confusion target, passed filters, evidence tier and sample weight. Internet sources were used to understand terminology; no public text was copied and auto-labeled.

## Leakage and 0–1-real policy

Generation uses frozen repeat-0/fold-0 real TRAIN only. Validation, calibration, temporal and sealed-test contents are not generation seeds and remain REAL ONLY. Zero-real classes are marked `SYNTHETIC_ONLY` and capped at weight 0.05; one-real classes use their single anchor, are marked `SINGLE_REAL_ANCHOR`, and are capped at 0.10. These rows do not count as independent real evidence.

## Downstream result

- Initial structured grid: `{synthetic.get('status', '—')}`; baseline `{json.dumps(synthetic.get('baseline', {}), ensure_ascii=False)}`; best `{json.dumps(synthetic.get('best', {}), ensure_ascii=False)}`.
- Family-specific real-validation experiments: `{json.dumps(synthetic_family_summary, ensure_ascii=False)}`.

Synthetic data survives for a model family only if it improves REAL validation macro-F1 without collapsing rare or worst-class F1. The chosen ratio and effective weight are frozen before the sealed holdout is opened.
""",
    )

    retrieval_lines = []
    for path in sorted((ART / "retrieval").glob("*.json")):
        item = load(path) or {}
        metrics = item.get("sealed_test", {})
        retrieval_lines.append(
            f"| {item.get('model_id', path.stem)} | {item.get('selected_dense_weight', '—')} | {metrics.get('mrr_at_10', '—')} | {metrics.get('recall_at_5', '—')} | {metrics.get('recall_at_10', '—')} | {metrics.get('recall_at_20', '—')} | {metrics.get('ndcg_at_10_proxy', '—')} |"
        )
    write(
        "RETRIEVAL_V4.md",
        f"""
# Retrieval V4

Weights and reranker cutoffs are tuned on retrieval-development queries; the table is the sealed retrieval test. Query self-hits and normalized-description groups are excluded.

| Encoder | Dense weight | MRR@10 | Recall@5 | Recall@10 | Recall@20 | nDCG@10 proxy |
|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(retrieval_lines)}

Qwen3-Reranker-0.6B is accepted only if it improves development MRR. `outputs/retrieval_human_review_v4.csv` contains representative pairs with blank 0–3 human labels; no labels were fabricated.

Scaling recipe: exact matrix + lexical search around 2k; benchmark exact vs HNSW around 20k; FAISS/Qdrant HNSW or IVF around 100k; Qdrant HNSW or FAISS IVF/PQ/HNSW around 1M.
""",
    )

    route_metrics = routing.get("metrics", {})
    write(
        "PRODUCTION_ARCHITECTURE_V4.md",
        f"""
# Production Architecture V4

## Decision path

Registration fields → active local category champion → internal OOD gate → known category or `UNKNOWN_NEW_ISSUE` → independent routing model → similar real cases → hierarchical SLA analytics.

- Normal inference always returns a final system decision; no mandatory human-review branch.
- The model and encoders load once. A v3 artifact remains as a tested failure fallback.
- Model provenance is returned by `/api/tickets/analyze` and exposed in system status.
- Routing comparison: `{json.dumps(route_metrics, ensure_ascii=False)}`.
- OOD test: `{json.dumps(ood.get('test', {}), ensure_ascii=False)}`. Weak OOD recall is a known limitation; UNKNOWN is deliberately conservative.
- Local specialist: `{json.dumps(specialist.get('metrics', {}), ensure_ascii=False)}`.

## Future retraining

Imports are cumulative/upserted. At 10k/100k+ tickets, final confirmed outcomes form a new dataset version; a new dataset hash creates a new frozen protocol. Challengers train on accumulated registration-time fields, compare with ACTIVE_CHAMPION using grouped and temporal tests, and are activated only after passing quality and latency gates. Registry states are ACTIVE_CHAMPION, PREVIOUS_CHAMPION and CHALLENGERS.

Monitoring stores necessary aggregates: class frequency, confidence/OOD rate, embedding drift, model disagreement and UNKNOWN clusters. Raw text is not duplicated merely for monitoring.
""",
    )


if __name__ == "__main__":
    main()
