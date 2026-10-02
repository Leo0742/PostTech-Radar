from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

from app.core.config import DATABASE_PATH  # noqa: E402
from app.ml.v4.evaluation import classification_metrics, safe_registration_row  # noqa: E402
from app.services.data_service import load_rows  # noqa: E402


def _predictions(view: str) -> tuple[dict[str, dict[str, str]], dict[str, str]]:
    models: dict[str, dict[str, str]] = {}
    truth: dict[str, str] = {}
    for path in sorted((PROJECT_ROOT / "artifacts/gpu_research_v4/embeddings").glob(f"*-{view}.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        for row in payload.get("predictions", {}).get("embedding_metadata_lr", []):
            request_id = str(row["request_id"])
            models.setdefault(payload["candidate_id"], {})[request_id] = str(row["prediction"])
            truth[request_id] = str(row["truth"])
    return models, truth


def _extract_label(output: str, candidates: list[str]) -> str | None:
    try:
        payload = json.loads(re.search(r"\{.*?\}", output, flags=re.DOTALL).group(0))  # type: ignore[union-attr]
        value = str(payload.get("category") or "")
        if value in candidates:
            return value
    except (AttributeError, json.JSONDecodeError):
        pass
    return next((label for label in candidates if label in output), None)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="Qwen/Qwen3-8B")
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import torch
    from huggingface_hub import model_info
    from transformers import AutoModelForCausalLM, AutoTokenizer

    models, truth = _predictions("top15")
    stack = json.loads(
        (PROJECT_ROOT / "artifacts/gpu_research_v4/stacking/top15.json").read_text(encoding="utf-8")
    )
    baseline = {str(row["request_id"]): str(row["stack"]) for row in stack["predictions"]}
    common = sorted(set.intersection(*(set(values) for values in models.values())))
    hard_ids = [
        item for item in common if len({values[item] for values in models.values()}) > 1
    ][: args.limit]
    rows = {str(row["request_id"]): row for row in load_rows(DATABASE_PATH)}
    info = model_info(args.model_id)
    revision = str(info.sha)
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, revision=revision, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        revision=revision,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        trust_remote_code=False,
    )
    records = []
    torch.cuda.reset_peak_memory_stats()
    for request_id in hard_ids:
        votes = [values[request_id] for values in models.values()]
        candidates = [label for label, _ in Counter(votes).most_common()]
        row = safe_registration_row(rows[request_id])
        prompt = (
            "Ты локальный классификатор Service Desk. Выбери ровно одну категорию только из списка. "
            "Не придумывай категорий. Верни только JSON {\"category\": \"точное имя\", "
            "\"rationale\": \"до 12 слов\"}.\n"
            f"Кандидаты: {json.dumps(candidates, ensure_ascii=False)}\n"
            f"Регистрационные поля: {json.dumps(row, ensure_ascii=False)}"
        )
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        started = time.perf_counter()
        with torch.inference_mode():
            generated = model.generate(**inputs, max_new_tokens=80, do_sample=False)
        latency = time.perf_counter() - started
        output = tokenizer.decode(generated[0, inputs.input_ids.shape[1] :], skip_special_tokens=True)
        specialist = _extract_label(output, candidates) or baseline[request_id]
        records.append(
            {
                "request_id": request_id,
                "truth": truth[request_id],
                "baseline": baseline[request_id],
                "specialist": specialist,
                "candidates": candidates,
                "latency_seconds": round(latency, 4),
                "raw_output": output,
            }
        )
    labels = sorted(set(truth[item] for item in hard_ids))
    payload = {
        "candidate_id": "category/qwen3-8b-local-specialist/top15/v4",
        "status": "MEASURED_HARD_DISAGREEMENT_SUBSET",
        "model_id": args.model_id,
        "revision": revision,
        "license": str((info.card_data or {}).get("license") or "UNKNOWN"),
        "trust_remote_code": False,
        "rows": len(records),
        "selection": "base-model disagreement only; labels not used for gate",
        "peak_vram_bytes": int(torch.cuda.max_memory_allocated()),
        "latency_seconds_mean": round(sum(row["latency_seconds"] for row in records) / max(1, len(records)), 4),
        "metrics": {
            "stack_baseline": classification_metrics(
                [row["truth"] for row in records], [row["baseline"] for row in records], labels=labels
            ),
            "specialist": classification_metrics(
                [row["truth"] for row in records], [row["specialist"] for row in records], labels=labels
            ),
        },
        "predictions": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "metrics": payload["metrics"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
