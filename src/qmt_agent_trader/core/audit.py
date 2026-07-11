"""Append-only JSONL audit log."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from qmt_agent_trader.core.ids import shanghai_now_iso
from qmt_agent_trader.persistence.atomic_files import AtomicFileStore
from qmt_agent_trader.persistence.audit import AuditJsonlStore
from qmt_agent_trader.persistence.locks import LockManager


@dataclass(frozen=True)
class AuditEvent:
    event_type: str
    actor: str
    payload: dict[str, Any]
    created_at: str = field(default_factory=shanghai_now_iso)


class AuditLogger:
    def __init__(
        self,
        path: Path,
        *,
        atomic_store: AtomicFileStore | None = None,
        fsync: bool = True,
        rotation_bytes: int | None = None,
    ) -> None:
        self.path = path.expanduser().resolve()
        resolved_store = atomic_store or AtomicFileStore(LockManager(self.path.parent / ".locks"))
        self._store = AuditJsonlStore(
            self.path, resolved_store, fsync=fsync, rotation_bytes=rotation_bytes
        )

    def append(self, event_type: str, actor: str, payload: dict[str, Any]) -> AuditEvent:
        event = AuditEvent(event_type=event_type, actor=actor, payload=payload)
        self._store.append(event.__dict__)
        return event

    def read_latest(self, limit: int = 100) -> list[dict[str, Any]]:
        return self._store.read_records(limit=limit)
