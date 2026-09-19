from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

VALID_STATUSES = {"champion", "challenger", "archived"}


class ModelRegistry:
    def __init__(self, path: Path):
        self.path = path

    def _read(self) -> dict[str, Any]:
        if not self.path.exists():
            return {"schema_version": "1.0", "models": []}
        return json.loads(self.path.read_text(encoding="utf-8"))

    def _write(self, payload: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        temporary.replace(self.path)

    def get(self, version: str) -> dict[str, Any]:
        record = next((item for item in self._read()["models"] if item["version"] == version), None)
        if record is None:
            raise KeyError(version)
        return record

    def champion(self) -> dict[str, Any] | None:
        return next((item for item in self._read()["models"] if item["status"] == "champion"), None)

    def register(self, record: dict[str, Any]) -> dict[str, Any]:
        version = str(record.get("version") or "").strip()
        status = str(record.get("status") or "challenger")
        if not version:
            raise ValueError("version is required")
        if status not in VALID_STATUSES:
            raise ValueError(f"invalid status: {status}")
        payload = self._read()
        existing = next((item for item in payload["models"] if item["version"] == version), None)
        if existing is not None:
            if existing != record:
                raise ValueError(f"model version {version} is immutable")
            return existing
        if status == "champion" and any(item["status"] == "champion" for item in payload["models"]):
            raise ValueError("a champion already exists; register as challenger and promote explicitly")
        stored = {**record, "version": version, "status": status}
        stored.setdefault("registered_at", datetime.now(UTC).isoformat())
        payload["models"].append(stored)
        self._write(payload)
        return stored

    def promote(self, version: str) -> dict[str, Any]:
        payload = self._read()
        target = next((item for item in payload["models"] if item["version"] == version), None)
        if target is None:
            raise KeyError(version)
        if target["status"] == "archived":
            raise ValueError("archived model cannot be promoted directly")
        now = datetime.now(UTC).isoformat()
        for item in payload["models"]:
            if item["status"] == "champion" and item["version"] != version:
                item["status"] = "archived"
                item["archived_at"] = now
        target["status"] = "champion"
        target["promoted_at"] = now
        self._write(payload)
        return target
