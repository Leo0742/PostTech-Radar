from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = PROJECT_ROOT / "artifacts" / "gpu_research_v5" / "environment" / "environment.json"
PACKAGES = (
    "torch",
    "torchvision",
    "transformers",
    "sentence-transformers",
    "huggingface-hub",
    "scikit-learn",
    "numpy",
    "scipy",
    "catboost",
    "accelerate",
    "peft",
    "safetensors",
)


def _command(args: list[str]) -> dict[str, Any]:
    try:
        result = subprocess.run(args, capture_output=True, text=True, check=False, timeout=30)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"available": False, "error": f"{type(exc).__name__}: {exc}"}
    return {
        "available": result.returncode == 0,
        "returncode": result.returncode,
        "stdout": result.stdout.strip(),
        "stderr": result.stderr.strip(),
    }


def _package_versions() -> dict[str, str | None]:
    result: dict[str, str | None] = {}
    for package in PACKAGES:
        try:
            result[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            result[package] = None
    return result


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hash_model_snapshot(path: Path) -> dict[str, Any]:
    files = [item for item in sorted(Path(path).rglob("*")) if item.is_file()]
    manifest = []
    for item in files:
        manifest.append(
            {
                "path": item.relative_to(path).as_posix(),
                "size": item.stat().st_size,
                "sha256": _sha256_file(item),
            }
        )
    canonical = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {
        "snapshot_path": str(path),
        "files": len(manifest),
        "bytes": sum(item["size"] for item in manifest),
        "artifact_sha256": hashlib.sha256(canonical).hexdigest(),
        "manifest": manifest,
    }


def collect_environment(*, model_id: str | None = None, revision: str | None = None, hash_snapshot: bool = False) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "python": {
            "version": sys.version,
            "executable": sys.executable,
            "platform": platform.platform(),
            "machine": platform.machine(),
        },
        "packages": _package_versions(),
        "environment": {
            "HF_HOME": os.environ.get("HF_HOME"),
            "HF_HUB_CACHE": os.environ.get("HF_HUB_CACHE"),
            "TRANSFORMERS_CACHE": os.environ.get("TRANSFORMERS_CACHE"),
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
        "nvidia_smi": _command(["nvidia-smi", "-q"]),
        "nvidia_smi_query": _command(
            [
                "nvidia-smi",
                "--query-gpu=name,uuid,driver_version,memory.total,compute_cap",
                "--format=csv,noheader,nounits",
            ]
        ),
        "nvcc": _command(["nvcc", "--version"]),
    }
    try:
        import torch

        payload["torch_runtime"] = {
            "version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "cudnn_version": torch.backends.cudnn.version(),
            "device_count": torch.cuda.device_count(),
            "devices": [
                {
                    "index": index,
                    "name": torch.cuda.get_device_name(index),
                    "capability": list(torch.cuda.get_device_capability(index)),
                    "total_memory": int(torch.cuda.get_device_properties(index).total_memory),
                }
                for index in range(torch.cuda.device_count())
            ],
        }
    except Exception as exc:
        payload["torch_runtime"] = {"available": False, "error": f"{type(exc).__name__}: {exc}"}

    if model_id and revision:
        payload["model"] = {"id": model_id, "revision": revision}
        if hash_snapshot:
            from huggingface_hub import snapshot_download

            snapshot = Path(snapshot_download(model_id, revision=revision, local_files_only=True))
            payload["model_snapshot"] = hash_model_snapshot(snapshot)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Capture reproducibility metadata for PostTech Radar V5.1 GPU runs")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model-id")
    parser.add_argument("--revision")
    parser.add_argument("--hash-model-snapshot", action="store_true")
    args = parser.parse_args()

    payload = collect_environment(
        model_id=args.model_id,
        revision=args.revision,
        hash_snapshot=args.hash_model_snapshot,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "cuda_available": payload.get("torch_runtime", {}).get("cuda_available")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
