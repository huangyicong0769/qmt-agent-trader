"""Daily price-volume factors."""

from __future__ import annotations

import numpy as np
import pandas as pd

from qmt_agent_trader.data.bars import column_quality, is_column_usable_for_factor


def momentum(frame: pd.DataFrame, window: int) -> pd.Series:
    return frame.groupby("symbol")["close"].pct_change(window)


def reversal_5d(frame: pd.DataFrame) -> pd.Series:
    return -momentum(frame, 5)


def _grouped_rolling_std(
    values: pd.Series,
    symbols: pd.Series,
    window: int,
) -> pd.Series:
    return (
        values.groupby(symbols, sort=False)
        .rolling(window)
        .std()
        .reset_index(level=0, drop=True)
        .reindex(values.index)
    )


def volatility_20d(frame: pd.DataFrame) -> pd.Series:
    returns = frame.groupby("symbol", sort=False)["close"].pct_change()
    return _grouped_rolling_std(returns, frame["symbol"], 20)


def turnover_20d(frame: pd.DataFrame) -> pd.Series:
    if "turnover" not in frame.columns:
        raise ValueError("TURNOVER_NOT_REAL_OR_INSUFFICIENT: turnover column is missing")
    quality = column_quality(frame, "turnover")
    if not is_column_usable_for_factor(frame, "turnover"):
        raise ValueError(
            "TURNOVER_NOT_REAL_OR_INSUFFICIENT: "
            f"turnover column_quality={quality}"
        )
    turnover = pd.to_numeric(frame["turnover"], errors="coerce")
    non_null = turnover.notna().sum()
    non_zero = turnover.fillna(0).ne(0).sum()
    if non_null == 0 or non_zero == 0:
        raise ValueError(
            "TURNOVER_NOT_REAL_OR_INSUFFICIENT: turnover has no non-zero observations"
        )
    return turnover.groupby(frame["symbol"]).rolling(20).mean().reset_index(level=0, drop=True)


def amount_zscore_20d(frame: pd.DataFrame) -> pd.Series:
    amount = frame.groupby("symbol")["amount"]
    rolling_mean = amount.rolling(20).mean().reset_index(level=0, drop=True)
    rolling_std = amount.rolling(20).std().replace(0, np.nan).reset_index(level=0, drop=True)
    return (frame["amount"] - rolling_mean) / rolling_std
