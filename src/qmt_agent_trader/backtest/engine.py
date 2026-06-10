"""Daily backtest engine with explicit signal/execution date separation."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from qmt_agent_trader.backtest.constraints import TradeState
from qmt_agent_trader.backtest.execution import SimulatedFill, simulate_daily_fill
from qmt_agent_trader.core.types import Side


@dataclass(frozen=True)
class BacktestResult:
    fills: list[SimulatedFill]
    leakage_valid: bool


class DailyBacktestEngine:
    def run_one_signal(
        self, bars: pd.DataFrame, *, symbol: str, signal_date: str, side: Side, quantity: int
    ) -> BacktestResult:
        symbol_bars = (
            bars[bars["symbol"] == symbol].sort_values("trade_date").reset_index(drop=True)
        )
        signal_matches = symbol_bars.index[
            symbol_bars["trade_date"].astype(str) == signal_date
        ].tolist()
        if not signal_matches:
            raise ValueError("signal date not found")
        execution_index = signal_matches[0] + 1
        if execution_index >= len(symbol_bars):
            return BacktestResult(fills=[], leakage_valid=True)
        row = symbol_bars.iloc[execution_index]
        state = TradeState(
            suspended=bool(row.get("suspended", False)),
            limit_up=bool(row.get("limit_up", False)),
            limit_down=bool(row.get("limit_down", False)),
            st=bool(row.get("st", False)),
        )
        fill = simulate_daily_fill(
            symbol=symbol,
            side=side,
            quantity=quantity,
            price=float(row["open"]),
            trade_date=pd.to_datetime(row["trade_date"]).date(),
            state=state,
        )
        return BacktestResult(fills=[] if fill is None else [fill], leakage_valid=True)
