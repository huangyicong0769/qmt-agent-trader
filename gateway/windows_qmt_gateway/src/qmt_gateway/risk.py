"""Gateway-side risk checks."""

from __future__ import annotations


def precheck_order_plan(plan: dict[str, object]) -> list[str]:
    reasons: list[str] = []
    if plan.get("strategy_approval_status") != "APPROVED":
        reasons.append("strategy is not approved")
    approval = plan.get("approval")
    if not isinstance(approval, dict) or approval.get("status") != "APPROVED":
        reasons.append("order plan approval missing")
    risk_checks = plan.get("risk_checks")
    if not isinstance(risk_checks, dict) or risk_checks.get("status") != "PASSED":
        reasons.append("risk checks did not pass")
    if not plan.get("plan_hash"):
        reasons.append("plan hash missing")
    if not plan.get("idempotency_key"):
        reasons.append("idempotency key missing")
    return reasons


def live_gate_reasons(*, live_trading_enabled: bool, allow_order_endpoint: bool) -> list[str]:
    reasons: list[str] = []
    if not live_trading_enabled:
        reasons.append("LIVE_TRADING_ENABLED is false")
    if not allow_order_endpoint:
        reasons.append("ALLOW_ORDER_ENDPOINT is false")
    return reasons
