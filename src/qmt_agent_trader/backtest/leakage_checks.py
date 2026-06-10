"""Leakage checks for point-in-time backtests."""

from __future__ import annotations

from datetime import date

import pandas as pd

from qmt_agent_trader.core.errors import LeakageError


def assert_signal_before_execution(signal_date: date, execution_date: date) -> None:
    if execution_date <= signal_date:
        raise LeakageError("execution must occur after signal date")


def assert_financials_visible(
    frame: pd.DataFrame, as_of: date, announced_column: str = "announced_at"
) -> None:
    announced = pd.to_datetime(frame[announced_column]).dt.date
    if bool((announced > as_of).any()):
        raise LeakageError("financial data contains future announcement rows")


def leakage_report(
    valid: bool, checks: list[str], errors: list[str] | None = None
) -> dict[str, object]:
    return {"valid": valid, "checks": checks, "errors": errors or []}
