from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for import_path in (ROOT, ROOT / "backend"):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from scripts.finalize_customer_adaptation import load_all
from scripts.gpu_customer_finetune import Config, _encode, _free_model, _train_encoder

OUT = ROOT / "models/customer_2026-09-21"
SEED = 20260921 + 9000
QWEN = Config("qwen_r16_mnrl_s80", 16, 2e-5, 80, "mnrl", 256, 8)
MINILM = Config("minilm_supcon_s120", 0, 2e-5, 120, "supcon", 256, 6)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    old_rows, new_rows, *_ = load_all()
    rows = old_rows + new_rows
    customer_ids = {str(row["request_id"]) for row in new_rows}
    OUT.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, object] = {
        "training_rows": len(rows),
        "historical_rows": len(old_rows),
        "customer_rows": len(new_rows),
        "lockbox_accessed": False,
    }

    qwen, qwen_stats = _train_encoder("qwen", QWEN, rows, customer_ids, SEED)
    qwen.eval()
    qwen_embeddings = _encode(qwen, "qwen", rows, 6)
    qwen_npz = OUT / "qwen_final_embeddings.npz"
    np.savez_compressed(
        qwen_npz,
        embeddings=qwen_embeddings,
        request_ids=np.asarray([str(row["request_id"]) for row in rows]),
        labels=np.asarray([str(row["category"]) for row in rows]),
    )
    adapter_dir = OUT / "qwen_r16_mnrl_s80_adapter"
    peft_model = qwen[0].auto_model
    peft_model.save_pretrained(adapter_dir)
    if hasattr(qwen, "tokenizer"):
        qwen.tokenizer.save_pretrained(adapter_dir)
    merged_dir = OUT / "qwen_r16_mnrl_s80_merged"
    qwen[0].auto_model = peft_model.merge_and_unload()
    qwen.save(str(merged_dir))
    manifest["qwen"] = {
        "config": QWEN.__dict__,
        "train": qwen_stats,
        "adapter_dir": str(adapter_dir.relative_to(ROOT)),
        "merged_dir": str(merged_dir.relative_to(ROOT)),
        "embeddings": str(qwen_npz.relative_to(ROOT)),
        "embeddings_sha256": sha256(qwen_npz),
    }
    _free_model(qwen)

    minilm, minilm_stats = _train_encoder("minilm", MINILM, rows, customer_ids, SEED + 1)
    minilm.eval()
    minilm_embeddings = _encode(minilm, "minilm", rows, 48)
    minilm_npz = OUT / "minilm_final_embeddings.npz"
    np.savez_compressed(
        minilm_npz,
        embeddings=minilm_embeddings,
        request_ids=np.asarray([str(row["request_id"]) for row in rows]),
        labels=np.asarray([str(row["category"]) for row in rows]),
    )
    minilm_dir = OUT / "minilm_supcon_s120"
    minilm.save(str(minilm_dir))
    manifest["minilm"] = {
        "config": MINILM.__dict__,
        "train": minilm_stats,
        "model_dir": str(minilm_dir.relative_to(ROOT)),
        "embeddings": str(minilm_npz.relative_to(ROOT)),
        "embeddings_sha256": sha256(minilm_npz),
    }
    _free_model(minilm)

    path = OUT / "training_manifest.json"
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
