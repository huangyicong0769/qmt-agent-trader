"""Audit log API routes."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi import APIRouter

from qmt_agent_trader.core.config import get_settings
from qmt_agent_trader.web.schemas import AuditSummary

router = APIRouter()


@router.get("/tool-calls", response_model=list[AuditSummary])
async def list_tool_calls(limit: int = 100) -> list[AuditSummary]:
    return _read_audit(limit=limit)


@router.get("/permission-denials", response_model=list[AuditSummary])
async def list_permission_denials(limit: int = 100) -> list[AuditSummary]:
    return [entry for entry in _read_audit(limit=limit * 3) if entry.status == "permission_denied"][
        :limit
    ]


@router.get("/errors", response_model=list[AuditSummary])
async def list_errors(limit: int = 100) -> list[AuditSummary]:
    return [entry for entry in _read_audit(limit=limit * 3) if entry.status == "error"][:limit]


def _read_audit(limit: int) -> list[AuditSummary]:
    entries: list[AuditSummary] = []
    for path in _audit_paths():
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                entries.append(_summary(payload))
    return sorted(entries, key=lambda entry: entry.timestamp, reverse=True)[:limit]


def _audit_paths() -> list[Path]:
    settings = get_settings()
    paths = sorted((settings.resolved_log_dir / "audit").glob("*.jsonl"))
    data_audit = settings.resolved_data_dir / "audit"
    if data_audit.exists():
        paths.extend(sorted(data_audit.glob("*.jsonl")))
    return paths


def _summary(payload: dict[str, Any]) -> AuditSummary:
    return AuditSummary(
        timestamp=str(payload.get("timestamp", "")),
        run_id=str(payload.get("run_id", "")),
        experiment_id=_optional_str(payload.get("experiment_id")),
        tool_name=str(payload.get("tool_name", "")),
        permission=str(payload.get("permission", "")),
        requested_by_llm=bool(payload.get("requested_by_llm", False)),
        input_hash=str(payload.get("input_hash", "")),
        output_hash=str(payload.get("output_hash", "")),
        status=str(payload.get("status", "")),
        error_message=_optional_str(payload.get("error_message")),
        duration_ms=int(payload.get("duration_ms", 0)),
    )


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    return str(value)
