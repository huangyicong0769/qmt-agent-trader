"""Backtest broker simulator."""

from __future__ import annotations

from dataclasses import dataclass, field

from qmt_agent_trader.backtest.execution import SimulatedFill
from qmt_agent_trader.core.types import Side


@dataclass
class BrokerSim:
    cash: float
    positions: dict[str, int] = field(default_factory=dict)

    def apply_fill(self, fill: SimulatedFill) -> None:
        notional = fill.quantity * fill.price
        if fill.side == Side.BUY:
            self.cash -= notional + fill.cost
            self.positions[fill.symbol] = self.positions.get(fill.symbol, 0) + fill.quantity
        else:
            self.cash += notional - fill.cost
            self.positions[fill.symbol] = self.positions.get(fill.symbol, 0) - fill.quantity
