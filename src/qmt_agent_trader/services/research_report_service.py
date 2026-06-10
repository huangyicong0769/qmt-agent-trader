"""Research artifact persistence for agent-generated evidence."""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

from qmt_agent_trader.core.ids import new_id, shanghai_now_iso


def save_research_report(
    reports_dir: Path,
    *,
    artifact_type: str,
    title: str,
    payload: dict[str, object],
    metadata: dict[str, object] | None = None,
    agent_notes: str | None = None,
    infrastructure_requests: list[str] | None = None,
) -> dict[str, object]:
    """Persist an immutable research artifact and return a compact receipt."""
    run_id = new_id("research")
    reports_dir.mkdir(parents=True, exist_ok=True)
    path = reports_dir / f"{run_id}.json"
    record = {
        "run_id": run_id,
        "created_at": shanghai_now_iso(),
        "artifact_type": artifact_type,
        "title": title,
        "research_only": True,
        "approval_status": "NOT_REQUESTED",
        "live_trading_allowed": False,
        "decision_boundary": {
            "can_support_review": True,
            "can_approve_strategy": False,
            "can_generate_live_order_plan": False,
            "requires_human_approval": True,
        },
        "metadata": metadata or {},
        "summary": payload.get("summary", {}),
        "payload": payload,
        "agent_notes": agent_notes,
        "infrastructure_requests": _normalize_requests(infrastructure_requests),
    }
    path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "status": "saved",
        "run_id": run_id,
        "path": str(path),
        "artifact_type": artifact_type,
        "research_only": True,
        "approval_status": "NOT_REQUESTED",
        "live_trading_allowed": False,
        "summary": record["summary"],
        "infrastructure_requests": record["infrastructure_requests"],
    }


def compare_research_reports(reports_dir: Path, *, limit: int = 10) -> dict[str, object]:
    """Return compact summaries of recent research artifacts."""
    if not reports_dir.exists():
        return {"status": "empty", "runs": [], "infrastructure_requests": []}
    paths = sorted(
        reports_dir.glob("research_*.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    runs = [_summarize_record(_load_json_object(path), path) for path in paths[:limit]]
    return {
        "status": "compared" if runs else "empty",
        "runs": runs,
        "infrastructure_requests": _collect_infrastructure_requests(runs),
    }


def _load_json_object(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"research report is not a JSON object: {path}")
    return cast(dict[str, object], value)


def _summarize_record(record: dict[str, object], path: Path) -> dict[str, object]:
    return {
        "run_id": record.get("run_id"),
        "created_at": record.get("created_at"),
        "artifact_type": record.get("artifact_type"),
        "title": record.get("title"),
        "research_only": record.get("research_only"),
        "approval_status": record.get("approval_status"),
        "live_trading_allowed": record.get("live_trading_allowed"),
        "metadata": record.get("metadata", {}),
        "summary": record.get("summary", {}),
        "agent_notes": record.get("agent_notes"),
        "infrastructure_requests": record.get("infrastructure_requests", []),
        "path": str(path),
    }


def _collect_infrastructure_requests(runs: list[dict[str, object]]) -> list[str]:
    requests: list[str] = []
    for run in runs:
        value = run.get("infrastructure_requests", [])
        if not isinstance(value, list):
            continue
        requests.extend(str(item) for item in value if str(item).strip())
    return requests


def _normalize_requests(requests: list[str] | None) -> list[str]:
    return [request.strip() for request in requests or [] if request.strip()]
