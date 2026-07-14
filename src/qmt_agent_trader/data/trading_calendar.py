"""Canonical expected trading sessions for backtest integrity checks."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta

import pandas as pd

from qmt_agent_trader.backtest.errors import BacktestDataIntegrityError
from qmt_agent_trader.data.storage import DataLake


@dataclass(frozen=True)
class TradingSessionWindow:
    warmup_dates: tuple[date, ...]
    expected_dates: tuple[date, ...]

    @property
    def panel_start(self) -> date:
        if self.warmup_dates:
            return self.warmup_dates[0]
        return self.expected_dates[0]


def _parse_boundary(value: str) -> date:
    for fmt in ("%Y%m%d", "%Y-%m-%d"):
        try:
            return datetime.strptime(str(value), fmt).date()
        except ValueError:
            continue
    raise BacktestDataIntegrityError(
        code="TRADING_CALENDAR_INVALID",
        message="calendar boundary is invalid",
        field="trade_cal",
        details={"value": value},
    )


def _natural_dates(start: date, end: date) -> tuple[date, ...]:
    if end < start:
        raise BacktestDataIntegrityError(
            code="TRADING_CALENDAR_INVALID",
            message="calendar end precedes start",
            field="trade_cal",
        )
    return tuple(start + timedelta(days=i) for i in range((end - start).days + 1))


def load_open_sessions(
    lake: DataLake,
    *,
    start: str,
    end: str,
    exchanges: tuple[str, ...] = ("SSE", "SZSE"),
) -> tuple[date, ...]:
    return load_session_window(
        lake,
        start=start,
        end=end,
        warmup_sessions=0,
        exchanges=exchanges,
    ).expected_dates


def load_session_window(
    lake: DataLake,
    *,
    start: str,
    end: str,
    warmup_sessions: int,
    exchanges: tuple[str, ...] = ("SSE", "SZSE"),
) -> TradingSessionWindow:
    if warmup_sessions < 0:
        raise ValueError("warmup_sessions must be non-negative")
    start_date = _parse_boundary(start)
    end_date = _parse_boundary(end)
    expected_dates = _natural_dates(start_date, end_date)
    states = _load_normalized_calendar_states(lake, exchanges=exchanges)
    missing_dates = [item for item in expected_dates if item not in states]
    if missing_dates:
        raise BacktestDataIntegrityError(
            code="TRADING_CALENDAR_PARTIAL_COVERAGE",
            message="trade calendar lacks evidence for natural dates in requested range",
            field="trade_cal",
            details={"missing_dates": [item.isoformat() for item in missing_dates]},
        )
    open_dates = tuple(day for day in expected_dates if states[day] == 1)
    prior_open = [day for day, is_open in states.items() if day < start_date and is_open == 1]
    warmup_dates = tuple(prior_open[-warmup_sessions:]) if warmup_sessions else ()
    if len(warmup_dates) != warmup_sessions:
        raise BacktestDataIntegrityError(
            code="INSUFFICIENT_FACTOR_WARMUP_HISTORY",
            message="trade calendar lacks enough prior open sessions for factor warm-up",
            field="trade_cal",
            details={
                "required_sessions": warmup_sessions,
                "available_sessions": len(prior_open),
                "start": start_date.isoformat(),
            },
        )
    if not open_dates:
        raise BacktestDataIntegrityError(
            code="TRADING_CALENDAR_EMPTY",
            message="trade calendar contains no open sessions for requested range",
            field="trade_cal",
            details={"start": start, "end": end},
        )
    return TradingSessionWindow(warmup_dates=warmup_dates, expected_dates=open_dates)


def _load_normalized_calendar_states(
    lake: DataLake,
    *,
    exchanges: tuple[str, ...],
) -> dict[date, int]:
    dataset = "tushare/trade_cal"
    if not lake.dataset_path("raw", dataset).exists():
        raise BacktestDataIntegrityError(
            code="TRADING_CALENDAR_NOT_READY",
            message="raw/tushare/trade_cal is required for backtest session validation",
            field="trade_cal",
            details={"exchanges": list(exchanges)},
        )
    frame = lake.read_parquet_filtered(
        "raw",
        dataset,
        columns=["exchange", "cal_date", "is_open"],
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
    normalized_dates = pd.to_datetime(
        frame["cal_date"].astype(str),
        format="mixed",
        errors="coerce",
    )
    normalized_states = pd.to_numeric(frame["is_open"], errors="coerce")
    invalid = normalized_dates.isna() | normalized_states.isna() | ~normalized_states.isin([0, 1])
    if invalid.any():
        raise BacktestDataIntegrityError(
            code="TRADING_CALENDAR_INVALID",
            message="trade calendar contains invalid date or state values",
            field="trade_cal",
            details={"invalid_row_count": int(invalid.sum())},
        )
    normalized = pd.DataFrame(
        {
            "cal_date": normalized_dates.dt.date,
            "is_open": normalized_states.astype(int),
        }
    )
    state_counts = normalized.groupby("cal_date")["is_open"].nunique()
    conflicting_dates = sorted(state_counts[state_counts > 1].index.tolist())
    if conflicting_dates:
        raise BacktestDataIntegrityError(
            code="TRADING_CALENDAR_CONFLICTING_STATE",
            message="trade calendar has conflicting exchange states for a date",
            field="trade_cal",
            details={"conflicting_dates": [item.isoformat() for item in conflicting_dates]},
        )
    states = normalized.groupby("cal_date", sort=True)["is_open"].first()
    return {day: int(state) for day, state in states.items()}
