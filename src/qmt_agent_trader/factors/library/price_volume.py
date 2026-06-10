"""Daily price-volume factors."""

from __future__ import annotations

import numpy as np
import pandas as pd


def momentum(frame: pd.DataFrame, window: int) -> pd.Series:
    return frame.groupby("symbol")["close"].pct_change(window)


def reversal_5d(frame: pd.DataFrame) -> pd.Series:
    return -momentum(frame, 5)


def volatility_20d(frame: pd.DataFrame) -> pd.Series:
    return (
        frame.groupby("symbol")["close"]
        .pct_change()
        .rolling(20)
        .std()
        .reset_index(level=0, drop=True)
    )


def turnover_20d(frame: pd.DataFrame) -> pd.Series:
    return frame.groupby("symbol")["turnover"].rolling(20).mean().reset_index(level=0, drop=True)


def amount_zscore_20d(frame: pd.DataFrame) -> pd.Series:
    amount = frame.groupby("symbol")["amount"]
    rolling_mean = amount.rolling(20).mean().reset_index(level=0, drop=True)
    rolling_std = amount.rolling(20).std().replace(0, np.nan).reset_index(level=0, drop=True)
    return (frame["amount"] - rolling_mean) / rolling_std
