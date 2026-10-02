from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class CandidateRegistry:
    def __init__(self, path: Path):
        self.path = path

    def _read(self) -> dict[str, Any]:
        if not self.path.exists():
            return {"schema_version": "3.0", "candidates": []}
        return json.loads(self.path.read_text(encoding="utf-8"))

    def append(self, record: dict[str, Any]) -> None:
        candidate_id = str(record.get("candidate_id") or "")
        if not candidate_id:
            raise ValueError("candidate_id is required")
        payload = self._read()
        existing = next((item for item in payload["candidates"] if item["candidate_id"] == candidate_id), None)
        if existing is not None:
            if existing != record:
                raise ValueError(f"Candidate registry is immutable for {candidate_id}")
            return
        payload["candidates"].append(record)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        temporary.replace(self.path)
