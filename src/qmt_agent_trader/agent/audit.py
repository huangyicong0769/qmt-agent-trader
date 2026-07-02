"""Agent audit logger — JSONL append-only trail for tool calls."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from hashlib import sha256
from pathlib import Path
from typing import Any


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
    _entry_cache: list[AuditEntry] = field(default_factory=list)

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
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(self._entry_dict(entry), ensure_ascii=False, default=str)
        with open(self.log_path, "a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    def flush(self) -> None:
        pass  # each append already flushes

    @staticmethod
    def _entry_dict(entry: AuditEntry) -> dict[str, object]:
        return {
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
        # Drop API keys / tokens that might slip into an error message.
        for fragment in ("sk-", "tushare", "hmac", "secret", "token"):
            if fragment in message.lower():
                return "[scrubbed]"
        return message

    @staticmethod
    def _now() -> str:
        import datetime

        return datetime.datetime.now(tz=datetime.UTC).isoformat()
