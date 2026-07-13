"""Typed fail-closed errors for research backtests."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BacktestDataIntegrityError(RuntimeError):
    code: str
    trade_date: str
    symbols: tuple[str, ...]
    field: str
    message: str

    def __str__(self) -> str:
        return (
            f"{self.code}: {self.message}; trade_date={self.trade_date}; "
            f"field={self.field}; symbols={list(self.symbols)}"
        )
