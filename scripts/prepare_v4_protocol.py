from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

from app.core.config import DATABASE_PATH  # noqa: E402
from app.ml.v4.protocol import audit_v4_protocol, build_v4_protocol, protocol_artifact_dir  # noqa: E402
from app.services.data_service import load_rows  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze the real-only PostTech Radar v4 protocol")
    parser.add_argument("--database", type=Path, default=DATABASE_PATH)
    parser.add_argument("--seed", type=int, default=20260917)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--folds", type=int, default=4)
    args = parser.parse_args()
    rows = [row for row in load_rows(args.database) if str(row.get("description") or "").strip()]
    protocol = build_v4_protocol(rows, seed=args.seed, repeats=args.repeats, n_splits=args.folds)
    payload = protocol.to_dict() | {
        "audit": audit_v4_protocol(protocol),
        "sealed": True,
        "evaluation_content": "REAL_ONLY",
        "rare_class_policy": (
            "Classes with fewer than five independent groups remain in development OOF; "
            "they are not placed in calibration/final holdout because a train/test split is not identifiable."
        ),
    }
    output_dir = PROJECT_ROOT / protocol_artifact_dir(protocol)
    output = output_dir / "protocol.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    if output.exists() and output.read_text(encoding="utf-8") != serialized:
        raise SystemExit(f"Refusing to overwrite frozen protocol: {output}")
    output.write_text(serialized, encoding="utf-8")
    pointer = PROJECT_ROOT / "artifacts/gpu_research_v4/protocol.json"
    pointer.write_text(
        json.dumps(
            {
                "dataset_sha256": protocol.dataset_sha256,
                "split_sha256": protocol.split_sha256,
                "protocol_path": str(output.relative_to(PROJECT_ROOT)),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output": str(output), "audit": payload["audit"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
