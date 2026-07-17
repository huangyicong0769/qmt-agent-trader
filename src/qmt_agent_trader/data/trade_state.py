"""Strict normalization for opening-auction stock execution state."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from qmt_agent_trader.backtest.errors import BacktestDataIntegrityError
from qmt_agent_trader.data.integrity import require_unique_keys, require_unique_symbol_dates

OPENING_TRADE_STATE_COLUMNS = (
    "suspended",
    "st",
    "limit_up_at_open",
    "limit_down_at_open",
)


def normalize_stock_opening_trade_state(
    bars: pd.DataFrame,
    *,
    suspend: pd.DataFrame,
    stk_limit: pd.DataFrame,
    namechange: pd.DataFrame,
) -> pd.DataFrame:
    limits = _normalize_stock_limits(stk_limit)
    suspend_events = _normalize_suspend_events(suspend)
    st_periods = _normalize_namechange_periods(namechange)
    result = _apply_opening_limits(bars, limits)
    suspended_keys = set(
        zip(suspend_events["symbol"], suspend_events["trade_date"], strict=False)
    )
    result["suspended"] = [
        (str(symbol), day) in suspended_keys
        for symbol, day in zip(
            result["symbol"].astype(str), result["trade_date"], strict=False
        )
    ]
    result["st"] = _historical_st_mask(result, st_periods)
    for column in OPENING_TRADE_STATE_COLUMNS:
        if result[column].isna().any():
            raise BacktestDataIntegrityError(
                code="UNKNOWN_EXECUTION_STATE",
                message="opening execution state contains unknown values",
                field=column,
            )
        result[column] = result[column].astype(bool)
    result.attrs["column_quality"] = bars.attrs.get("column_quality", {})
    result.attrs["trade_state_quality"] = {
        "asset_type": "stock",
        "execution_time": "open",
        "suspended": {"source": "raw/tushare/suspend_d", "complete": True},
        "st": {"source": "raw/tushare/namechange", "complete": True},
        "limit_up_at_open": {
            "source": "raw/tushare/stk_limit",
            "complete": True,
        },
        "limit_down_at_open": {
            "source": "raw/tushare/stk_limit",
            "complete": True,
        },
    }
    return result


def _apply_opening_limits(
    bars: pd.DataFrame,
    limits: pd.DataFrame,
) -> pd.DataFrame:
    result = bars.copy()
    _require_columns(result, ("symbol", "trade_date", "open"), field="bars")
    result["symbol"] = result["symbol"].astype(str)
    result["trade_date"] = _coerce_dates(result["trade_date"], field="bars.trade_date")
    require_unique_symbol_dates(
        result,
        symbol_column="symbol",
        date_column="trade_date",
        code="DUPLICATE_SYMBOL_DATE_BAR",
        field="bars",
    )
    result = result.merge(
        limits,
        on=["symbol", "trade_date"],
        how="left",
        validate="one_to_one",
    )
    missing = result["up_limit"].isna() | result["down_limit"].isna()
    if missing.any():
        raise BacktestDataIntegrityError(
            code="TRADE_STATE_PARTIAL_COVERAGE",
            message="limit source does not cover every executable bar",
            field="raw/tushare/stk_limit",
            symbols=tuple(sorted(result.loc[missing, "symbol"].astype(str).unique())),
            details={"field": "limit_up_down", "missing_key_count": int(missing.sum())},
        )
    opening_prices = pd.to_numeric(result["open"], errors="coerce")
    if opening_prices.isna().any():
        raise BacktestDataIntegrityError(
            code="INVALID_REQUIRED_PRICE",
            message="opening price is required for opening-limit state",
            field="open",
            details={"invalid_row_count": int(opening_prices.isna().sum())},
        )
    tolerance = 1e-6
    result["limit_up_at_open"] = opening_prices >= result["up_limit"] - tolerance
    result["limit_down_at_open"] = opening_prices <= result["down_limit"] + tolerance
    return result.drop(columns=["up_limit", "down_limit"])


def normalize_etf_opening_trade_state(
    bars: pd.DataFrame,
    *,
    stk_limit: pd.DataFrame,
) -> pd.DataFrame:
    limits = _normalize_stock_limits(stk_limit)
    result = _apply_opening_limits(bars, limits)
    result["suspended"] = False
    result["st"] = False
    for column in OPENING_TRADE_STATE_COLUMNS:
        result[column] = result[column].astype(bool)
    result.attrs["column_quality"] = bars.attrs.get("column_quality", {})
    result.attrs["trade_state_quality"] = {
        "asset_type": "etf",
        "execution_time": "open",
        "suspended": {
            "source": "presence_of_valid_fund_daily_bar",
            "complete": True,
        },
        "st": {"source": "not_applicable_for_etf", "complete": True},
        "limit_up_at_open": {
            "source": "raw/tushare/stk_limit",
            "complete": True,
        },
        "limit_down_at_open": {
            "source": "raw/tushare/stk_limit",
            "complete": True,
        },
    }
    return result


def _normalize_stock_limits(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.rename(columns={"ts_code": "symbol"}).copy()
    _require_columns(
        data,
        ("symbol", "trade_date", "up_limit", "down_limit"),
        field="raw/tushare/stk_limit",
    )
    data["symbol"] = data["symbol"].astype(str)
    data["trade_date"] = _coerce_dates(
        data["trade_date"], field="raw/tushare/stk_limit.trade_date"
    )
    require_unique_symbol_dates(
        data,
        symbol_column="symbol",
        date_column="trade_date",
        code="DUPLICATE_TRADE_STATE_INPUT",
        field="raw/tushare/stk_limit",
    )
    for column in ("up_limit", "down_limit"):
        data[column] = pd.to_numeric(data[column], errors="coerce")
    invalid = (
        ~np.isfinite(data["up_limit"])
        | ~np.isfinite(data["down_limit"])
        | (data["up_limit"] <= 0)
        | (data["down_limit"] <= 0)
        | (data["down_limit"] >= data["up_limit"])
    )
    if invalid.any():
        raise BacktestDataIntegrityError(
            code="INVALID_TRADE_STATE_SOURCE",
            message="stock limit source contains invalid prices",
            field="raw/tushare/stk_limit",
            symbols=tuple(sorted(data.loc[invalid, "symbol"].astype(str).unique())),
            details={"invalid_row_count": int(invalid.sum())},
        )
    return data[["symbol", "trade_date", "up_limit", "down_limit"]]


def _normalize_suspend_events(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.rename(columns={"ts_code": "symbol"}).copy()
    _require_columns(
        data,
        ("symbol", "trade_date"),
        field="raw/tushare/suspend_d",
    )
    data["symbol"] = data["symbol"].astype(str)
    data["trade_date"] = _coerce_dates(
        data["trade_date"], field="raw/tushare/suspend_d.trade_date"
    )
    require_unique_symbol_dates(
        data,
        symbol_column="symbol",
        date_column="trade_date",
        code="DUPLICATE_TRADE_STATE_INPUT",
        field="raw/tushare/suspend_d",
    )
    return data[["symbol", "trade_date"]]


def _normalize_namechange_periods(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.rename(columns={"ts_code": "symbol"}).copy()
    _require_columns(
        data,
        ("symbol", "name", "start_date", "end_date"),
        field="raw/tushare/namechange",
    )
    data["symbol"] = data["symbol"].astype(str)
    data["start_date"] = _coerce_dates(
        data["start_date"], field="raw/tushare/namechange.start_date"
    )
    data["end_date"] = _coerce_dates(
        data["end_date"],
        field="raw/tushare/namechange.end_date",
        missing_value=date(2099, 12, 31),
    )
    invalid = data["end_date"] < data["start_date"]
    if invalid.any():
        raise BacktestDataIntegrityError(
            code="INVALID_TRADE_STATE_SOURCE",
            message="namechange source contains an invalid interval",
            field="raw/tushare/namechange",
            symbols=tuple(sorted(data.loc[invalid, "symbol"].astype(str).unique())),
            details={"invalid_row_count": int(invalid.sum())},
        )
    require_unique_keys(
        data,
        keys=("symbol", "name", "start_date", "end_date"),
        code="DUPLICATE_TRADE_STATE_INPUT",
        field="raw/tushare/namechange",
    )
    return data[["symbol", "name", "start_date", "end_date"]]


def _historical_st_mask(bars: pd.DataFrame, periods: pd.DataFrame) -> pd.Series:
    result = pd.Series(False, index=bars.index, dtype=bool)
    st_periods = periods[
        periods["name"].astype(str).str.contains("ST", case=False, na=False)
    ]
    for row in st_periods.itertuples(index=False):
        result |= (
            bars["symbol"].eq(str(row.symbol))
            & bars["trade_date"].ge(row.start_date)
            & bars["trade_date"].le(row.end_date)
        )
    return result


def _coerce_dates(
    values: pd.Series,
    *,
    field: str,
    missing_value: date | None = None,
) -> pd.Series:
    text = values.astype("string").str.strip()
    missing = values.isna() | text.isin(["", "NaT", "None", "nan", "<NA>"])
    parsed = pd.to_datetime(text.where(~missing), format="mixed", errors="coerce")
    invalid = parsed.isna() & ~missing
    if invalid.any() or (missing.any() and missing_value is None):
        raise BacktestDataIntegrityError(
            code="INVALID_TRADE_STATE_SOURCE",
            message="trade-state source contains an invalid date",
            field=field,
            details={"invalid_row_count": int((invalid | missing).sum())},
        )
    result = parsed.dt.date
    if missing_value is not None:
        result = result.where(~missing, missing_value)
    return result


def _require_columns(
    frame: pd.DataFrame,
    required: tuple[str, ...],
    *,
    field: str,
) -> None:
    missing = sorted(set(required).difference(frame.columns))
    if missing:
        raise BacktestDataIntegrityError(
            code="INVALID_TRADE_STATE_SOURCE",
            message="trade-state source is missing required columns",
            field=field,
            details={"missing_columns": missing},
        )
