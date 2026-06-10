"""Execution simulation."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from qmt_agent_trader.backtest.commission import calculate_cost
from qmt_agent_trader.backtest.constraints import TradeState, is_tradeable
from qmt_agent_trader.core.types import Side


@dataclass(frozen=True)
class SimulatedFill:
    symbol: str
    side: Side
    quantity: int
    price: float
    trade_date: date
    cost: float


def simulate_daily_fill(
    *,
    symbol: str,
    side: Side,
    quantity: int,
    price: float,
    trade_date: date,
    state: TradeState,
) -> SimulatedFill | None:
    if not is_tradeable(side, state):
        return None
    notional = quantity * price
    return SimulatedFill(
        symbol=symbol,
        side=side,
        quantity=quantity,
        price=price,
        trade_date=trade_date,
        cost=calculate_cost(notional, side),
    )
