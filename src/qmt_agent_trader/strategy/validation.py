"""Validation helpers for generated strategy outputs."""

from __future__ import annotations

import math
from typing import Any

import pandas as pd

from qmt_agent_trader.strategy.signal import TargetPortfolio

_FORBIDDEN_OUTPUT_FIELDS = {
    "broker",
    "gateway",
    "xtquant",
    "submit_order",
    "live",
    "account_secret",
}


def validate_signals(
    frame: pd.DataFrame,
    *,
    long_only: bool = True,
    max_single_position_pct: float = 1.0,
    max_abs_sum: float = 1.0,
) -> list[str]:
    issues: list[str] = []
    if frame.empty:
        return issues
    for column in _FORBIDDEN_OUTPUT_FIELDS.intersection(frame.columns):
        issues.append(f"forbidden output field: {column}")
    if "symbol" not in frame.columns:
        issues.append("missing required column: symbol")
        return issues
    if "target_weight" not in frame.columns:
        issues.append("missing required column: target_weight")
        return issues
    if frame["symbol"].duplicated().any():
        issues.append("duplicate symbols are not allowed")
    weights = pd.to_numeric(frame["target_weight"], errors="coerce")
    if weights.isna().any() or not weights.map(math.isfinite).all():
        issues.append("target_weight must be finite numeric values")
        return issues
    if long_only and (weights < 0).any():
        issues.append("long-only strategy cannot produce negative target_weight")
    if (weights.abs() > max_single_position_pct).any():
        issues.append("target_weight exceeds max_single_position_pct")
    if float(weights.abs().sum()) > max_abs_sum + 1e-9:
        issues.append("sum of absolute target weights exceeds allowed maximum")
    return issues


def validate_target_portfolio(portfolio: TargetPortfolio) -> list[str]:
    issues: list[str] = []
    symbols = [position.symbol for position in portfolio.positions]
    if len(symbols) != len(set(symbols)):
        issues.append("duplicate symbols are not allowed")
    weight_sum = portfolio.cash_weight
    for position in portfolio.positions:
        if not math.isfinite(position.target_weight):
            issues.append(f"{position.symbol} target_weight must be finite")
        if position.target_weight < 0:
            issues.append(f"{position.symbol} target_weight cannot be negative")
        weight_sum += position.target_weight
    if weight_sum > 1.0 + 1e-9:
        issues.append("target portfolio weights plus cash exceed 1.0")
    return issues


def assert_no_duplicate_symbols(frame: pd.DataFrame) -> None:
    if "symbol" not in frame.columns:
        raise ValueError("missing required column: symbol")
    if frame["symbol"].duplicated().any():
        raise ValueError("duplicate symbols are not allowed")


def assert_weights_valid(
    frame: pd.DataFrame,
    *,
    long_only: bool = True,
    max_abs_sum: float = 1.0,
) -> None:
    issues = validate_signals(frame, long_only=long_only, max_abs_sum=max_abs_sum)
    if issues:
        raise ValueError("; ".join(issues))


def signal_issues_as_payload(issues: list[str]) -> dict[str, Any]:
    return {"status": "PASSED" if not issues else "FAILED", "issues": issues}
