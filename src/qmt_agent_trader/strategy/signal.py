"""Signal models."""

from __future__ import annotations

from pydantic import BaseModel, Field


class StrategySignal(BaseModel):
    symbol: str
    signal_date: str
    score: float | None = None
    target_weight: float
    reason: str = ""
    metadata: dict[str, object] = Field(default_factory=dict)


class TargetPosition(BaseModel):
    symbol: str
    target_weight: float
    target_quantity: int | None = None
    target_notional: float | None = None
    reason: str = ""


class TargetPortfolio(BaseModel):
    strategy_id: str
    as_of_date: str
    positions: list[TargetPosition]
    cash_weight: float = 0.0
    metadata: dict[str, object] = Field(default_factory=dict)


class Signal(BaseModel):
    symbol: str
    target_weight: float
    reason: str

    def to_strategy_signal(self, *, signal_date: str) -> StrategySignal:
        return StrategySignal(
            symbol=self.symbol,
            signal_date=signal_date,
            target_weight=self.target_weight,
            reason=self.reason,
        )
