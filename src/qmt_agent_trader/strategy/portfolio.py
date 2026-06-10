"""Portfolio construction helpers."""

from __future__ import annotations

from qmt_agent_trader.strategy.signal import Signal


def equal_weight_top_n(symbols: list[str], n: int) -> list[Signal]:
    selected = symbols[:n]
    if not selected:
        return []
    weight = 1 / len(selected)
    return [
        Signal(symbol=symbol, target_weight=weight, reason="equal_weight_top_n")
        for symbol in selected
    ]
