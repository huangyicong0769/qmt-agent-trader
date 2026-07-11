"""Research artifact persistence for agent-generated evidence."""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

from qmt_agent_trader.core.ids import new_id, shanghai_now_iso
from qmt_agent_trader.persistence.artifacts import ArtifactMetadata, artifact_store_for_root
from qmt_agent_trader.persistence.locks import LockManager


def save_research_report(
    reports_dir: Path,
    *,
    artifact_type: str,
    title: str,
    payload: dict[str, object],
    metadata: dict[str, object] | None = None,
    agent_notes: str | None = None,
    infrastructure_requests: list[str] | None = None,
    lock_manager: LockManager | None = None,
) -> dict[str, object]:
    """Persist an immutable research artifact and return a compact receipt."""
    run_id = new_id("research")
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
        "review_gate": evaluate_research_gate(artifact_type, payload),
        "payload": payload,
        "agent_notes": agent_notes,
        "infrastructure_requests": _normalize_requests(infrastructure_requests),
    }
    content = json.dumps(record, ensure_ascii=False, indent=2).encode("utf-8")
    receipt = artifact_store_for_root(reports_dir, lock_manager=lock_manager).create(
        f"{run_id}.json",
        content,
        metadata=ArtifactMetadata(
            artifact_id=run_id,
            artifact_type=artifact_type,
            producer="services.research_report_service.save_research_report",
            related_run_id=run_id,
            related_strategy_id=_metadata_id(metadata, "strategy_id"),
            related_factor_id=_metadata_id(metadata, "factor_id", "factor_name"),
        ),
    )
    path = receipt.path
    return {
        "status": "saved",
        "run_id": run_id,
        "path": str(path),
        "artifact_type": artifact_type,
        "research_only": True,
        "approval_status": "NOT_REQUESTED",
        "live_trading_allowed": False,
        "summary": record["summary"],
        "review_gate": record["review_gate"],
        "infrastructure_requests": record["infrastructure_requests"],
        "storage_status": {
            "status": "VERIFIED",
            "component": "artifact_store",
            "reason": None,
            "warnings": [],
            "repair_action": None,
        },
    }


def _metadata_id(metadata: dict[str, object] | None, *keys: str) -> str | None:
    values = metadata or {}
    for key in keys:
        value = values.get(key)
        if value is not None and str(value):
            return str(value)
    return None


def compare_research_reports(
    reports_dir: Path, *, limit: int = 10, lock_manager: LockManager | None = None
) -> dict[str, object]:
    """Return compact summaries of recent research artifacts."""
    if not reports_dir.exists():
        return {"status": "empty", "runs": [], "infrastructure_requests": []}
    paths = sorted(
        reports_dir.glob("research_*.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    runs = [
        _summarize_record(_load_governed_report(path, reports_dir, lock_manager=lock_manager), path)
        for path in paths[:limit]
    ]
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


def _load_governed_report(
    path: Path, reports_dir: Path, *, lock_manager: LockManager | None = None
) -> dict[str, object]:
    store = artifact_store_for_root(reports_dir, lock_manager=lock_manager)
    run_id = path.stem
    if store.manifest_path_for(run_id).exists():
        raw = store.read_verified(run_id, expected_relative_path=path.name)
        value = json.loads(raw)
    else:

        def validate_legacy(content: bytes) -> bool:
            candidate = json.loads(content)
            return isinstance(candidate, dict) and str(candidate.get("run_id")) == run_id

        receipt = store.adopt(
            path.name,
            metadata=ArtifactMetadata(
                artifact_id=run_id,
                artifact_type="legacy_research_report",
                producer="services.research_report_service.legacy_adoption",
                related_run_id=run_id,
            ),
            validator=validate_legacy,
        )
        value = json.loads(receipt.content)
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
        "review_gate": record.get("review_gate", {}),
        "agent_notes": record.get("agent_notes"),
        "infrastructure_requests": record.get("infrastructure_requests", []),
        "path": str(path),
    }


def evaluate_research_gate(
    artifact_type: str,
    payload: dict[str, object],
) -> dict[str, object]:
    """Evaluate whether a research artifact has enough evidence for human review."""
    if artifact_type != "factor_rank_sensitivity":
        return {
            "status": "NOT_EVALUATED",
            "checks": [],
            "required_before_review": ["add a research gate for this artifact type"],
        }
    summary = _payload_summary(payload)
    scenarios = _payload_scenarios(payload)
    checks = [
        _gate_check(
            "min_scenarios",
            _summary_int(summary, "scenario_count") >= 4,
            "run at least four robustness scenarios",
        ),
        _gate_check(
            "diagnostics_pass",
            _summary_float(summary, "pass_ratio") >= 1.0,
            "all sensitivity scenarios must pass diagnostics",
            failed=True,
        ),
        _gate_check(
            "cost_stress_present",
            any(_scenario_float(scenario, "cost_multiplier", 1.0) > 1.0 for scenario in scenarios),
            "include at least one higher transaction-cost scenario",
        ),
        _gate_check(
            "slippage_stress_present",
            any(_scenario_float(scenario, "slippage_bps", 0.0) > 0.0 for scenario in scenarios),
            "include at least one non-zero slippage scenario",
        ),
        _gate_check(
            "delay_stress_present",
            any(_scenario_int(scenario, "execution_delay_days", 1) > 1 for scenario in scenarios),
            "include at least one delayed-execution scenario",
        ),
        _gate_check(
            "parameter_grid_present",
            _distinct_scenario_values(scenarios, "top_n") > 1
            or _distinct_scenario_values(scenarios, "max_single_position_pct") > 1,
            "compare at least two top_n or position-cap settings",
        ),
    ]
    missing = [str(check["message"]) for check in checks if check["status"] in {"WARN", "FAILED"}]
    return {
        "status": _gate_status(checks),
        "checks": checks,
        "required_before_review": missing,
    }


def _payload_summary(payload: dict[str, object]) -> dict[str, object]:
    value = payload.get("summary", {})
    return value if isinstance(value, dict) else {}


def _payload_scenarios(payload: dict[str, object]) -> list[dict[str, object]]:
    value = payload.get("runs", [])
    if not isinstance(value, list):
        return []
    scenarios: list[dict[str, object]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        scenario = item.get("scenario", {})
        if isinstance(scenario, dict):
            scenarios.append(scenario)
    return scenarios


def _gate_check(
    name: str,
    passed: bool,
    message: str,
    *,
    failed: bool = False,
) -> dict[str, object]:
    if passed:
        return {"name": name, "status": "PASSED", "message": message}
    return {"name": name, "status": "FAILED" if failed else "WARN", "message": message}


def _gate_status(checks: list[dict[str, object]]) -> str:
    statuses = {str(check["status"]) for check in checks}
    if "FAILED" in statuses:
        return "FAILED"
    if "WARN" in statuses:
        return "INSUFFICIENT_EVIDENCE"
    return "PASSED"


def _summary_int(summary: dict[str, object], key: str) -> int:
    value = summary.get(key, 0)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    try:
        return int(value) if isinstance(value, str) else 0
    except (TypeError, ValueError):
        return 0


def _summary_float(summary: dict[str, object], key: str) -> float:
    value = summary.get(key, 0.0)
    if isinstance(value, int | float):
        return float(value)
    try:
        return float(value) if isinstance(value, str) else 0.0
    except (TypeError, ValueError):
        return 0.0


def _scenario_float(scenario: dict[str, object], key: str, default: float) -> float:
    value = scenario.get(key, default)
    if isinstance(value, int | float):
        return float(value)
    try:
        return float(value) if isinstance(value, str) else default
    except (TypeError, ValueError):
        return default


def _scenario_int(scenario: dict[str, object], key: str, default: int) -> int:
    value = scenario.get(key, default)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    try:
        return int(value) if isinstance(value, str) else default
    except (TypeError, ValueError):
        return default


def _distinct_scenario_values(scenarios: list[dict[str, object]], key: str) -> int:
    return len({str(scenario.get(key)) for scenario in scenarios if scenario.get(key) is not None})


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
