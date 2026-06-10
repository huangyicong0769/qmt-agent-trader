"""Backtest metrics."""

from __future__ import annotations


def total_return(start_value: float, end_value: float) -> float:
    return end_value / start_value - 1
