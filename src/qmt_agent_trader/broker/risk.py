"""Mac-side order plan risk checks."""

from __future__ import annotations

from qmt_agent_trader.broker.order_plan import OrderPlan, RiskCheckResult, RiskChecks
from qmt_agent_trader.core.types import RiskStatus


def check_max_order_value(plan: OrderPlan, max_order_value: float) -> RiskCheckResult:
    for order in plan.orders:
        if order.limit_price is not None and order.limit_price * order.quantity > max_order_value:
            return RiskCheckResult(
                name="max_order_value",
                status=RiskStatus.FAILED,
                message=f"{order.symbol} exceeds max_order_value",
            )
    return RiskCheckResult(name="max_order_value", status=RiskStatus.PASSED)


def run_order_plan_risk_checks(plan: OrderPlan, max_order_value: float = 100000) -> RiskChecks:
    checks = (check_max_order_value(plan, max_order_value=max_order_value),)
    status = (
        RiskStatus.PASSED
        if all(check.status == RiskStatus.PASSED for check in checks)
        else RiskStatus.FAILED
    )
    return RiskChecks(status=status, checks=checks)
