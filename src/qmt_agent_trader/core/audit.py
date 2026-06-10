"""Append-only JSONL audit log."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from qmt_agent_trader.core.ids import shanghai_now_iso


@dataclass(frozen=True)
class AuditEvent:
    event_type: str
    actor: str
    payload: dict[str, Any]
    created_at: str = field(default_factory=shanghai_now_iso)


class AuditLogger:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, event_type: str, actor: str, payload: dict[str, Any]) -> AuditEvent:
        event = AuditEvent(event_type=event_type, actor=actor, payload=payload)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event.__dict__, ensure_ascii=False, sort_keys=True))
            handle.write("\n")
        return event

    def read_latest(self, limit: int = 100) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        lines = self.path.read_text(encoding="utf-8").splitlines()[-limit:]
        return [json.loads(line) for line in lines if line.strip()]
