from __future__ import annotations

import argparse

from huggingface_hub import snapshot_download
from project_setup import QWEN_MODEL_ID, QWEN_REVISION, project_root


def main() -> None:
    parser = argparse.ArgumentParser(description="Скачать базовые веса модели с Hugging Face")
    parser.add_argument("--force", action="store_true", help="проверить и докачать файлы повторно")
    args = parser.parse_args()

    target = project_root() / "models" / "encoders" / "Qwen3-Embedding-4B"
    ready = target / "model-00002-of-00002.safetensors"
    if ready.exists() and not args.force:
        print(f"Qwen уже скачан: {target}")
        return

    print("Скачиваем Qwen3-Embedding-4B. Нужно около 8 ГБ свободного места.")
    snapshot_download(
        repo_id=QWEN_MODEL_ID,
        revision=QWEN_REVISION,
        local_dir=target,
    )
    if not ready.exists():
        raise RuntimeError("Hugging Face download finished without the expected Qwen weight files")
    print(f"Qwen готов: {target}")
    print(f"Зафиксированная ревизия: {QWEN_REVISION}")


if __name__ == "__main__":
    main()
