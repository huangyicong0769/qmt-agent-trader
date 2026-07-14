"""Typed fail-closed errors for research backtests."""

from __future__ import annotations

from typing import Any


class BacktestIntegrityError(RuntimeError):
    """Base class for known fail-closed research-backtest integrity violations."""

    def __init__(
        self,
        *,
        code: str,
        message: str,
        trade_date: str | None = None,
        symbols: tuple[str, ...] = (),
        field: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.trade_date = trade_date
        self.symbols = symbols
        self.field = field
        self.details = dict(details or {})

    def as_dict(self) -> dict[str, object]:
        return {
            "code": self.code,
            "message": self.message,
            "trade_date": self.trade_date,
            "symbols": list(self.symbols),
            "field": self.field,
            "details": self.details,
        }

    def __str__(self) -> str:
        return (
            f"{self.code}: {self.message}; trade_date={self.trade_date}; "
            f"field={self.field}; symbols={list(self.symbols)}; details={self.details}"
        )


class BacktestDataIntegrityError(BacktestIntegrityError):
    """Required market or calendar data is absent or invalid."""


class BacktestUniverseIntegrityError(BacktestDataIntegrityError):
    """Point-in-time universe membership cannot be resolved."""


class BacktestAccountingError(BacktestDataIntegrityError):
    """The simulated ledger violates an accounting invariant."""
