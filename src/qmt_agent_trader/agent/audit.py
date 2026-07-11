"""Agent audit logger — JSONL append-only trail for tool calls."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from hashlib import sha256
from pathlib import Path
from typing import Any

from qmt_agent_trader.persistence.atomic_files import AtomicFileStore
from qmt_agent_trader.persistence.audit import AuditJsonlStore
from qmt_agent_trader.persistence.locks import LockManager


@dataclass
class AuditEntry:
    timestamp: str
    run_id: str
    session_id: str | None
    experiment_id: str | None
    tool_name: str
    permission: str
    requested_by_llm: bool
    call_mode: str
    input_hash: str
    output_hash: str
    status: str  # "ok" | "permission_denied" | "error"
    error_message: str | None
    duration_ms: int
    execution_status: str = "UNKNOWN"
    domain_status: str = "UNKNOWN"
    evidence_status: str = "UNKNOWN"
    recommendation_status: str = "UNKNOWN"
    raw_status: str | None = None
    diagnostic_status: str | None = None
    blockers: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    next_repair_tool: str | None = None
    output_data: dict[str, Any] | None = None


@dataclass
class AuditLogger:
    log_path: Path
    atomic_store: AtomicFileStore | None = None
    fsync: bool = True
    rotation_bytes: int | None = None
    _entry_cache: list[AuditEntry] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.log_path = self.log_path.expanduser().resolve()
        if self.atomic_store is None:
            from qmt_agent_trader.core.config import get_settings
            from qmt_agent_trader.persistence.paths import PersistencePaths

            manager = LockManager(PersistencePaths.from_settings(get_settings()).locks_root)
            store = AtomicFileStore(manager)
        else:
            store = self.atomic_store
        self._store = AuditJsonlStore(
            self.log_path, store, fsync=self.fsync, rotation_bytes=self.rotation_bytes
        )

    def append(
        self,
        tool_name: str,
        run_id: str,
        *,
        session_id: str | None = None,
        experiment_id: str | None = None,
        permission: str = "READ_ONLY",
        requested_by_llm: bool = True,
        call_mode: str = "AUTONOMOUS_AGENT",
        input_data: dict[str, Any] | None = None,
        output_data: dict[str, Any] | None = None,
        status: str = "ok",
        error_message: str | None = None,
        duration_ms: int = 0,
        execution_status: str = "UNKNOWN",
        domain_status: str = "UNKNOWN",
        evidence_status: str = "UNKNOWN",
        recommendation_status: str = "UNKNOWN",
        raw_status: str | None = None,
        diagnostic_status: str | None = None,
        blockers: list[str] | None = None,
        warnings: list[str] | None = None,
        next_repair_tool: str | None = None,
    ) -> None:
        entry = AuditEntry(
            timestamp=self._now(),
            run_id=run_id,
            session_id=session_id,
            experiment_id=experiment_id,
            tool_name=tool_name,
            permission=permission,
            requested_by_llm=requested_by_llm,
            call_mode=call_mode,
            input_hash=self._safe_hash(input_data),
            output_hash=self._safe_hash(output_data),
            status=status,
            error_message=self._scrub_error(error_message),
            duration_ms=duration_ms,
            execution_status=execution_status,
            domain_status=domain_status,
            evidence_status=evidence_status,
            recommendation_status=recommendation_status,
            raw_status=raw_status,
            diagnostic_status=diagnostic_status,
            blockers=blockers or [],
            warnings=warnings or [],
            next_repair_tool=next_repair_tool,
            output_data=output_data,
        )
        self._entry_cache.append(entry)
        self._flush_one(entry)

    def _flush_one(self, entry: AuditEntry) -> None:
        self._store.append(self._entry_dict(entry))

    def flush(self) -> None:
        pass  # each append already flushes

    @staticmethod
    def _entry_dict(entry: AuditEntry) -> dict[str, object]:
        value = {
            "timestamp": entry.timestamp,
            "run_id": entry.run_id,
            "session_id": entry.session_id,
            "experiment_id": entry.experiment_id,
            "tool_name": entry.tool_name,
            "permission": entry.permission,
            "requested_by_llm": entry.requested_by_llm,
            "call_mode": entry.call_mode,
            "input_hash": entry.input_hash,
            "output_hash": entry.output_hash,
            "status": entry.status,
            "error_message": entry.error_message,
            "duration_ms": entry.duration_ms,
            "execution_status": entry.execution_status,
            "domain_status": entry.domain_status,
            "evidence_status": entry.evidence_status,
            "recommendation_status": entry.recommendation_status,
            "raw_status": entry.raw_status,
            "diagnostic_status": entry.diagnostic_status,
            "blockers": entry.blockers,
            "warnings": entry.warnings,
            "next_repair_tool": entry.next_repair_tool,
            "output_data": entry.output_data,
        }
        scrubbed = _scrub_value(value)
        if not isinstance(scrubbed, dict):  # pragma: no cover - root shape is fixed above
            raise TypeError("audit entry must remain an object after secret scrubbing")
        return scrubbed

    @staticmethod
    def _safe_hash(data: object) -> str:
        if data is None:
            return "none"
        try:
            canonical = json.dumps(data, ensure_ascii=False, sort_keys=True, default=str)
            return sha256(canonical.encode("utf-8")).hexdigest()[:16]
        except (TypeError, ValueError):
            return "unhashable"

    @staticmethod
    def _scrub_error(message: str | None) -> str | None:
        if message is None:
            return None
        return "[scrubbed]" if _contains_credential(message) else message

    @staticmethod
    def _now() -> str:
        import datetime

        return datetime.datetime.now(tz=datetime.UTC).isoformat()


def scrub_sensitive(value: Any) -> Any:
    """Apply the audit subsystem's recursive credential scrub to public diagnostics."""
    return _scrub_value(value)


_SAFE_TELEMETRY_NAMES = frozenset(
    {
        "completiontokens",
        "prompttokens",
        "tokenbudget",
        "tokencount",
        "tokenusage",
        "totaltokens",
    }
)
_CREDENTIAL_NAME_SUFFIXES = (
    "accesskey",
    "accesstoken",
    "apikey",
    "apisecret",
    "authtoken",
    "authorization",
    "bearer",
    "clientsecret",
    "hmacsecret",
    "password",
    "refreshtoken",
    "secret",
    "token",
)
_CREDENTIAL_PATTERNS = (
    re.compile(r"\bsk-[A-Za-z0-9_-]{4,}\b", re.IGNORECASE),
    re.compile(r"\bBearer\s+[A-Za-z0-9._~+/=-]+", re.IGNORECASE),
)
_ASSIGNMENT_PATTERN = re.compile(
    r"(?P<prefix>(?P<key_quote>[\"']?)(?P<name>[A-Za-z][A-Za-z0-9_.-]*)"
    r"(?P=key_quote)\s*[:=]\s*)"
    r"(?P<value>\"(?:\\.|[^\"])*\"|'(?:\\.|[^'])*'|[^\s,;}\]]+)"
)


def _scrub_value(value: Any, *, key: str = "") -> Any:
    if _is_credential_key(key):
        return "[scrubbed]"
    if isinstance(value, dict):
        return {
            str(item_key): _scrub_value(item, key=str(item_key)) for item_key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_scrub_value(item) for item in value]
    if isinstance(value, str):
        if any(pattern.search(value) for pattern in _CREDENTIAL_PATTERNS):
            return "[scrubbed]"
        return _redact_assignments(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _contains_credential(value: str) -> bool:
    return any(pattern.search(value) for pattern in _CREDENTIAL_PATTERNS) or (
        _redact_assignments(value) != value
    )


def _redact_assignments(value: str) -> str:
    def replace(match: re.Match[str]) -> str:
        if not _is_credential_key(match.group("name")):
            return match.group(0)
        assigned = match.group("value")
        if assigned[:1] in {'"', "'"} and assigned[-1:] == assigned[:1]:
            redacted = f"{assigned[0]}[scrubbed]{assigned[0]}"
        else:
            redacted = "[scrubbed]"
        return f"{match.group('prefix')}{redacted}"

    return _ASSIGNMENT_PATTERN.sub(replace, value)


def _is_credential_key(key: str) -> bool:
    canonical = re.sub(r"[^a-z0-9]+", "", key.lower())
    if canonical in _SAFE_TELEMETRY_NAMES:
        return False
    return any(canonical.endswith(suffix) for suffix in _CREDENTIAL_NAME_SUFFIXES)
