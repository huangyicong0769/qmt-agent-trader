"""Slippage models."""

from __future__ import annotations

from qmt_agent_trader.core.types import Side


def fixed_bps_slippage(price: float, side: Side, bps: float = 5.0) -> float:
    adjustment = price * bps / 10000
    return price + adjustment if side == Side.BUY else price - adjustment
