"""Build dry-run order plans from strategy target portfolios."""

from __future__ import annotations

import pandas as pd

from qmt_agent_trader.broker.order import Order
from qmt_agent_trader.broker.order_plan import OrderPlan, OrderPlanApproval, RiskChecks
from qmt_agent_trader.broker.risk import run_order_plan_risk_checks
from qmt_agent_trader.core.types import ApprovalStatus, OrderType, Side
from qmt_agent_trader.strategy.portfolio import round_lot_quantity
from qmt_agent_trader.strategy.signal import TargetPortfolio


def build_order_plan_from_target_portfolio(
    *,
    portfolio: TargetPortfolio,
    current_positions: dict[str, int],
    prices: pd.DataFrame,
    strategy_version: str,
    account_id_hash: str,
    dry_run: bool = True,
    strategy_approval_status: ApprovalStatus = ApprovalStatus.APPROVED,
    equity: float = 1_000_000,
) -> OrderPlan:
    if not dry_run and strategy_approval_status != ApprovalStatus.APPROVED:
        raise ValueError("non-paper order plans require approved strategy")
    price_map = _price_map(prices)
    orders: list[Order] = []
    for target in portfolio.positions:
        price = price_map.get(target.symbol)
        if price is None or price <= 0:
            continue
        desired_quantity = target.target_quantity
        if desired_quantity is None:
            desired_quantity = round_lot_quantity(int(equity * target.target_weight / price))
        current_quantity = int(current_positions.get(target.symbol, 0))
        delta = desired_quantity - current_quantity
        quantity = round_lot_quantity(abs(delta))
        if quantity <= 0:
            continue
        orders.append(
            Order(
                symbol=target.symbol,
                side=Side.BUY if delta > 0 else Side.SELL,
                quantity=quantity,
                order_type=OrderType.LIMIT,
                limit_price=price,
                reason=target.reason or "strategy target rebalance",
            )
        )
    plan = OrderPlan(
        strategy_id=portfolio.strategy_id,
        strategy_version=strategy_version,
        strategy_approval_status=strategy_approval_status,
        account_id_hash=account_id_hash,
        dry_run=dry_run,
        orders=tuple(orders),
        risk_checks=RiskChecks.passed([]),
        approval=OrderPlanApproval(
            required=not dry_run,
            status=ApprovalStatus.APPROVED if dry_run else ApprovalStatus.REVIEW_REQUIRED,
            approved_by="paper" if dry_run else None,
        ),
    )
    risk_checks = run_order_plan_risk_checks(plan)
    return plan.model_copy(update={"risk_checks": risk_checks})


def _price_map(prices: pd.DataFrame) -> dict[str, float]:
    if prices.empty:
        return {}
    price_column = "price" if "price" in prices.columns else "close"
    if not {"symbol", price_column}.issubset(prices.columns):
        raise ValueError("prices must include symbol and price/close")
    latest = prices.drop_duplicates("symbol", keep="last")
    return {
        str(row.symbol): float(getattr(row, price_column))
        for row in latest.itertuples(index=False)
    }
