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
    experiment_id: str | None
    tool_name: str
    permission: str
    requested_by_llm: bool
    input_hash: str
    output_hash: str
    status: str  # "ok" | "permission_denied" | "error"
    error_message: str | None
    duration_ms: int


@dataclass
class AuditLogger:
    log_path: Path
    _entry_cache: list[AuditEntry] = field(default_factory=list)

    def append(
        self,
        tool_name: str,
        run_id: str,
        *,
        experiment_id: str | None = None,
        permission: str = "READ_ONLY",
        requested_by_llm: bool = True,
        input_data: dict[str, Any] | None = None,
        output_data: dict[str, Any] | None = None,
        status: str = "ok",
        error_message: str | None = None,
        duration_ms: int = 0,
    ) -> None:
        entry = AuditEntry(
            timestamp=self._now(),
            run_id=run_id,
            experiment_id=experiment_id,
            tool_name=tool_name,
            permission=permission,
            requested_by_llm=requested_by_llm,
            input_hash=self._safe_hash(input_data),
            output_hash=self._safe_hash(output_data),
            status=status,
            error_message=self._scrub_error(error_message),
            duration_ms=duration_ms,
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
            "experiment_id": entry.experiment_id,
            "tool_name": entry.tool_name,
            "permission": entry.permission,
            "requested_by_llm": entry.requested_by_llm,
            "input_hash": entry.input_hash,
            "output_hash": entry.output_hash,
            "status": entry.status,
            "error_message": entry.error_message,
            "duration_ms": entry.duration_ms,
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
