"""Factor evaluation context assembled from bars and PIT fundamentals."""

from __future__ import annotations

from datetime import date

import pandas as pd

from qmt_agent_trader.data.bars import load_daily_bars
from qmt_agent_trader.data.fundamentals import load_fundamentals_asof, parse_date
from qmt_agent_trader.data.storage import DataLake


def load_factor_context(
    lake: DataLake,
    *,
    as_of_date: str | date,
    symbols: list[str] | None = None,
    include_bars: bool = True,
    include_fundamentals: bool = True,
) -> pd.DataFrame:
    as_of = parse_date(as_of_date)
    frames: list[pd.DataFrame] = []
    if include_bars:
        bars = load_daily_bars(lake, end=as_of, symbols=symbols)
        if not bars.empty:
            latest_bars = (
                bars.sort_values(["symbol", "trade_date"])
                .groupby("symbol", as_index=False)
                .tail(1)
                .reset_index(drop=True)
            )
            frames.append(latest_bars)
    if include_fundamentals:
        fundamentals = load_fundamentals_asof(
            lake,
            as_of_date=as_of,
            symbols=symbols,
        )
        if not fundamentals.empty:
            fundamentals = fundamentals.drop(columns=["trade_date"], errors="ignore")
            frames.append(fundamentals)
    if not frames:
        return pd.DataFrame()

    result = frames[0]
    for frame in frames[1:]:
        result = result.merge(frame, on="symbol", how="outer", suffixes=("", "_fundamental"))
    if "trade_date" not in result.columns:
        result["trade_date"] = as_of
    result["as_of_date"] = as_of
    return result.sort_values("symbol").reset_index(drop=True)
