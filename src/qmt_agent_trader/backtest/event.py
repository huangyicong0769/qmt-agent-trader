"""Backtest events."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date


@dataclass(frozen=True)
class TradeEvent:
    signal_date: date
    execution_date: date
    symbol: str
    side: str
    quantity: int
    price: float
