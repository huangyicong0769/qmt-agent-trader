"""Process-safe JSONL audit persistence and read-only verification."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from qmt_agent_trader.persistence.atomic_files import AtomicFileStore
from qmt_agent_trader.persistence.errors import StorageCorruptError


@dataclass(frozen=True)
class AuditCorruption:
    line_number: int
    reason: str


@dataclass
class AuditVerification:
    path: Path
    valid_records: int = 0
    tail_truncated: bool = False
    corruptions: list[AuditCorruption] = field(default_factory=list)

    @property
    def healthy(self) -> bool:
        return not self.tail_truncated and not self.corruptions


@dataclass(frozen=True)
class AuditReadResult:
    records: list[dict[str, Any]]
    verification: AuditVerification


def verify_audit_jsonl(path: Path) -> AuditVerification:
    return _read_audit_jsonl(path).verification


def _read_audit_jsonl(path: Path) -> AuditReadResult:
    result = AuditVerification(path=path)
    records: list[dict[str, Any]] = []
    paths = _audit_paths(path)
    if not paths:
        return AuditReadResult(records, result)
    raw_lines = [
        (source, raw)
        for source in paths
        for raw in source.read_bytes().splitlines(keepends=True)
    ]
    for index, (source, raw) in enumerate(raw_lines, start=1):
        try:
            value = json.loads(raw)
            if not isinstance(value, dict):
                raise ValueError("audit record must be an object")
            result.valid_records += 1
            records.append(value)
        except (json.JSONDecodeError, UnicodeDecodeError, ValueError) as exc:
            is_final = source == path and index == len(raw_lines)
            if is_final and not raw.endswith(b"\n"):
                result.tail_truncated = True
            else:
                result.corruptions.append(AuditCorruption(index, str(exc)))
    return AuditReadResult(records, result)


class AuditJsonlStore:
    def __init__(
        self,
        path: Path,
        atomic_store: AtomicFileStore,
        *,
        schema_version: int = 2,
        fsync: bool = True,
        rotation_bytes: int | None = None,
    ) -> None:
        self.path = path.expanduser().resolve()
        self.atomic_store = atomic_store
        self.schema_version = schema_version
        self.fsync = fsync
        self.rotation_bytes = rotation_bytes

    def append(self, record: dict[str, Any]) -> None:
        versioned = {**record, "schema_version": self.schema_version}
        self.atomic_store.rotate_and_append_jsonl(
            self.path,
            versioned,
            rotation_bytes=self.rotation_bytes,
            fsync=self.fsync,
            compact=False,
        )

    def verify(self) -> AuditVerification:
        with self.atomic_store.lock_manager.resource_lock(self.path):
            return _read_audit_jsonl(self.path).verification

    def read_records(self, *, limit: int | None = None) -> list[dict[str, Any]]:
        result = self.read_records_with_diagnostics(limit=limit)
        if not result.verification.healthy:
            raise StorageCorruptError(
                store_name="audit",
                path=self.path,
                operation="read_records",
                reason="audit stream contains truncated or corrupt records",
                suggested_repair="inspect the audit verification diagnostics",
            )
        return result.records

    def read_records_with_diagnostics(self, *, limit: int | None = None) -> AuditReadResult:
        with self.atomic_store.lock_manager.resource_lock(self.path):
            result = _read_audit_jsonl(self.path)
            selected = result.records[-limit:] if limit is not None else result.records
            return AuditReadResult(selected, result.verification)


def _audit_paths(path: Path) -> list[Path]:
    legacy = path.with_suffix(path.suffix + ".1")
    prefix = f"{path.name}."
    generations = sorted(
        (
            candidate
            for candidate in path.parent.glob(f"{path.name}.*")
            if candidate != legacy and candidate.name.removeprefix(prefix).isdigit()
        ),
        key=lambda candidate: int(candidate.name.removeprefix(prefix)),
    )
    return [
        *([legacy] if legacy.exists() else []),
        *generations,
        *([path] if path.exists() else []),
    ]
