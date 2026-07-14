"""Canonical expected trading sessions for backtest integrity checks."""

from __future__ import annotations

from datetime import date

import pandas as pd

from qmt_agent_trader.backtest.errors import BacktestDataIntegrityError
from qmt_agent_trader.data.storage import DataLake


def load_open_sessions(
    lake: DataLake,
    *,
    start: str,
    end: str,
    exchanges: tuple[str, ...] = ("SSE", "SZSE"),
) -> tuple[date, ...]:
    dataset = "tushare/trade_cal"
    if not lake.dataset_path("raw", dataset).exists():
        raise BacktestDataIntegrityError(
            code="TRADING_CALENDAR_NOT_READY",
            message="raw/tushare/trade_cal is required for backtest session validation",
            field="trade_cal",
            details={"start": start, "end": end, "exchanges": list(exchanges)},
        )
    frame = lake.read_parquet_filtered(
        "raw",
        dataset,
        columns=["exchange", "cal_date", "is_open"],
        start=start,
        end=end,
        date_column="cal_date",
    )
    required = {"cal_date", "is_open"}
    missing = required.difference(frame.columns)
    if missing:
        raise BacktestDataIntegrityError(
            code="TRADING_CALENDAR_INVALID",
            message="trade calendar is missing required columns",
            field="trade_cal",
            details={"missing_columns": sorted(missing)},
        )
    if "exchange" in frame.columns:
        frame = frame[frame["exchange"].astype(str).isin(exchanges)]
    frame = frame[pd.to_numeric(frame["is_open"], errors="coerce") == 1]
    dates = tuple(sorted(pd.to_datetime(frame["cal_date"].astype(str)).dt.date.unique()))
    if not dates:
        raise BacktestDataIntegrityError(
            code="TRADING_CALENDAR_EMPTY",
            message="trade calendar contains no open sessions for requested range",
            field="trade_cal",
            details={"start": start, "end": end},
        )
    return dates
