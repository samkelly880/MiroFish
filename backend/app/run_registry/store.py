"""
Thin run registry.

Stores `uploads/runs/<run_id>/manifest.json` that points at existing
project / simulation / report directories without replacing upstream storage.
"""

from __future__ import annotations

import json
import os
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from ..config import Config


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class RunRecord:
    run_id: str
    status: str = "created"
    created_at: str = field(default_factory=_utc_now)
    updated_at: str = field(default_factory=_utc_now)
    requirement: str = ""
    platform: str = "parallel"
    max_rounds: Optional[int] = None
    project_id: Optional[str] = None
    graph_id: Optional[str] = None
    simulation_id: Optional[str] = None
    report_id: Optional[str] = None
    error: Optional[str] = None
    artifacts: Dict[str, str] = field(default_factory=dict)
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RunRecord":
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        payload = {k: v for k, v in data.items() if k in known}
        return cls(**payload)


class RunRegistry:
    """Filesystem-backed index of CLI runs."""

    def __init__(self, root_dir: Optional[str] = None):
        self.root_dir = root_dir or os.path.join(Config.UPLOAD_FOLDER, "runs")
        os.makedirs(self.root_dir, exist_ok=True)

    def _run_dir(self, run_id: str) -> str:
        return os.path.join(self.root_dir, run_id)

    def manifest_path(self, run_id: str) -> str:
        return os.path.join(self._run_dir(run_id), "manifest.json")

    def create(
        self,
        *,
        requirement: str = "",
        platform: str = "parallel",
        max_rounds: Optional[int] = None,
        meta: Optional[Dict[str, Any]] = None,
        run_id: Optional[str] = None,
    ) -> RunRecord:
        run_id = run_id or f"run_{uuid.uuid4().hex[:12]}"
        record = RunRecord(
            run_id=run_id,
            requirement=requirement,
            platform=platform,
            max_rounds=max_rounds,
            meta=meta or {},
        )
        os.makedirs(self._run_dir(run_id), exist_ok=True)
        self.save(record)
        return record

    def save(self, record: RunRecord) -> None:
        record.updated_at = _utc_now()
        path = self.manifest_path(record.run_id)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(record.to_dict(), handle, ensure_ascii=False, indent=2)

    def get(self, run_id: str) -> Optional[RunRecord]:
        path = self.manifest_path(run_id)
        if not os.path.isfile(path):
            return None
        with open(path, "r", encoding="utf-8") as handle:
            return RunRecord.from_dict(json.load(handle))

    def update(self, run_id: str, **fields: Any) -> RunRecord:
        record = self.get(run_id)
        if record is None:
            raise KeyError(f"Unknown run_id: {run_id}")
        for key, value in fields.items():
            if key == "artifacts" and isinstance(value, dict):
                record.artifacts.update(value)
            elif key == "meta" and isinstance(value, dict):
                record.meta.update(value)
            elif hasattr(record, key):
                setattr(record, key, value)
        self.save(record)
        return record

    def list(self, limit: int = 50) -> List[RunRecord]:
        if not os.path.isdir(self.root_dir):
            return []
        records: List[RunRecord] = []
        for name in os.listdir(self.root_dir):
            record = self.get(name)
            if record is not None:
                records.append(record)
        records.sort(key=lambda r: r.created_at, reverse=True)
        return records[:limit]

    def artifact_path(self, run_id: str, *parts: str) -> str:
        path = os.path.join(self._run_dir(run_id), *parts)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        return path

    def write_artifact(self, run_id: str, relative_path: str, content: str) -> str:
        path = self.artifact_path(run_id, relative_path)
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(content)
        self.update(run_id, artifacts={relative_path: path})
        return path
