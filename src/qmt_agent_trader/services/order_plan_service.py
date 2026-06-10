"""Order plan generation service."""

from __future__ import annotations

from pathlib import Path

from qmt_agent_trader.broker.order import Order
from qmt_agent_trader.broker.order_plan import OrderPlan, OrderPlanApproval, RiskChecks
from qmt_agent_trader.core.types import ApprovalStatus, OrderType, Side


def build_sample_paper_order_plan(strategy_id: str) -> OrderPlan:
    return OrderPlan(
        strategy_id=strategy_id,
        strategy_version="1.0.0",
        strategy_approval_status=ApprovalStatus.APPROVED,
        account_id_hash="paper_account",
        dry_run=True,
        orders=(
            Order(
                symbol="000001.SZ",
                side=Side.BUY,
                quantity=100,
                order_type=OrderType.LIMIT,
                limit_price=10.0,
                reason="sample paper rebalance",
            ),
        ),
        risk_checks=RiskChecks.passed(["max_order_value", "cash_available"]),
        approval=OrderPlanApproval(status=ApprovalStatus.APPROVED, approved_by="human"),
    )


def save_order_plan(plan: OrderPlan, directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{plan.order_plan_id}.json"
    path.write_text(plan.model_dump_json(indent=2), encoding="utf-8")
    return path


def load_order_plan(identifier: str, directory: Path = Path("order_plans")) -> OrderPlan:
    path = Path(identifier)
    if not path.exists():
        path = directory / f"{identifier}.json"
    if not path.exists():
        raise ValueError(f"order plan not found: {identifier}")
    return OrderPlan.model_validate_json(path.read_text(encoding="utf-8"))
