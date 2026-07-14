"""Canonical bar readers backed by the data lake."""

from __future__ import annotations

from datetime import date, datetime

import pandas as pd

from qmt_agent_trader.backtest.errors import BacktestDataIntegrityError
from qmt_agent_trader.data.integrity import require_unique_symbol_dates
from qmt_agent_trader.data.storage import DataLake
from qmt_agent_trader.data.trade_state import normalize_stock_opening_trade_state

CANONICAL_BAR_COLUMNS = [
    "symbol",
    "trade_date",
    "asset_type",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "amount",
    "turnover",
    "suspended",
    "st",
    "limit_up_at_open",
    "limit_down_at_open",
]

_REQUIRED_TRADE_STATE_DATASETS = {
    "suspended": "tushare/suspend_d",
    "limit_up_down": "tushare/stk_limit",
    "st": "tushare/namechange",
}


def normalize_tushare_daily(frame: pd.DataFrame, *, asset_type: str | None = None) -> pd.DataFrame:
    resolved_asset_type = asset_type or "stock"
    if resolved_asset_type not in {"stock", "etf"}:
        raise ValueError(f"unsupported asset_type: {resolved_asset_type}")
    if frame.empty:
        empty = pd.DataFrame(columns=CANONICAL_BAR_COLUMNS)
        empty.attrs["column_quality"] = {}
        return empty

    data = frame.copy()
    column_quality: dict[str, dict[str, object]] = {}
    if "_empty" in data.columns:
        data = data.drop(columns=["_empty"])
    if data.empty:
        empty = pd.DataFrame(columns=CANONICAL_BAR_COLUMNS)
        empty.attrs["column_quality"] = {}
        return empty

    symbol_column = "ts_code" if "ts_code" in data.columns else "symbol"
    required = {symbol_column, "trade_date", "open", "high", "low", "close"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"tushare daily bars missing columns: {sorted(missing)}")
    require_unique_symbol_dates(
        data,
        symbol_column=symbol_column,
        date_column="trade_date",
        code="DUPLICATE_SYMBOL_DATE_BAR",
        field="raw_daily_bars",
    )

    rename_map = {"ts_code": "symbol", "vol": "volume"}
    data = data.rename(columns=rename_map)
    data["asset_type"] = resolved_asset_type
    data["trade_date"] = pd.to_datetime(data["trade_date"].astype(str), format="%Y%m%d").dt.date
    for column in ["volume", "amount"]:
        if column not in data.columns:
            data[column] = pd.NA
            column_quality[column] = {
                "source": "missing_from_raw",
                "imputed": True,
                "usable_for_factor": False,
            }
        else:
            column_quality[column] = {
                "source": "raw",
                "imputed": False,
                "usable_for_factor": True,
            }
    if "turnover" not in data.columns:
        data["turnover"] = pd.NA
        column_quality["turnover"] = {
            "source": "missing_from_raw",
            "imputed": True,
            "usable_for_factor": False,
        }
    else:
        column_quality["turnover"] = {
            "source": "raw",
            "imputed": False,
            "usable_for_factor": True,
        }
    for column in ["suspended", "st", "limit_up_at_open", "limit_down_at_open"]:
        if column not in data.columns:
            data[column] = pd.NA

    normalized = (
        data[CANONICAL_BAR_COLUMNS]
        .sort_values(["symbol", "trade_date"], kind="stable")
        .reset_index(drop=True)
    )
    normalized.attrs["column_quality"] = column_quality
    return normalized


def load_daily_bars(
    lake: DataLake,
    *,
    start: str | date | None = None,
    end: str | date | None = None,
    symbols: list[str] | None = None,
    include_trade_state: bool = True,
) -> pd.DataFrame:
    daily_columns = [
        "ts_code",
        "trade_date",
        "open",
        "high",
        "low",
        "close",
        "vol",
        "volume",
        "amount",
        "turnover",
    ]
    stock_raw = lake.read_parquet_filtered(
        "raw",
        "tushare/daily",
        columns=daily_columns,
        start=start,
        end=end,
        symbols=symbols,
    )
    etf_raw = lake.read_parquet_filtered(
        "raw",
        "tushare/fund_daily",
        columns=daily_columns,
        start=start,
        end=end,
        symbols=symbols,
    )
    normalized = [
        normalized_frame
        for frame, asset_type in ((stock_raw, "stock"), (etf_raw, "etf"))
        if not frame.empty
        for normalized_frame in [normalize_tushare_daily(frame, asset_type=asset_type)]
    ]
    bars = (
        pd.concat(normalized, ignore_index=True)
        if normalized
        else pd.DataFrame(columns=CANONICAL_BAR_COLUMNS)
    )
    bars.attrs["column_quality"] = _merge_column_quality(normalized)
    if bars.empty:
        return bars
    if not include_trade_state:
        result = bars.sort_values(["symbol", "trade_date"]).reset_index(drop=True)
        result.attrs["column_quality"] = bars.attrs.get("column_quality", {})
        return result

    etf_bars = bars[bars["asset_type"] == "etf"]
    if not etf_bars.empty:
        raise BacktestDataIntegrityError(
            code="UNSUPPORTED_ETF_TRADE_STATE_MODEL",
            message="ETF rows cannot use stock-only stk_limit evidence",
            field="trade_state",
            symbols=tuple(sorted(etf_bars["symbol"].astype(str).unique())),
        )

    _require_trade_state_sources(lake)
    suspend = lake.read_parquet_filtered(
        "raw",
        "tushare/suspend_d",
        columns=["ts_code", "trade_date", "suspend_type"],
        start=start,
        end=end,
        symbols=symbols,
    )
    stk_limit = lake.read_parquet_filtered(
        "raw",
        "tushare/stk_limit",
        columns=["ts_code", "trade_date", "up_limit", "down_limit"],
        start=start,
        end=end,
        symbols=symbols,
    )
    namechange = _filter_namechange_overlap(
        lake.read_parquet_filtered(
            "raw",
            "tushare/namechange",
            columns=["ts_code", "name", "start_date", "end_date"],
            symbols=symbols,
        ),
        start=start,
        end=end,
    )
    bars = enrich_trade_states(
        bars,
        suspend=suspend,
        stk_limit=stk_limit,
        namechange=namechange,
    )

    result = bars.sort_values(["symbol", "trade_date"]).reset_index(drop=True)
    result.attrs["column_quality"] = bars.attrs.get("column_quality", {})
    result.attrs["trade_state_quality"] = bars.attrs.get("trade_state_quality", {})
    return result


def _require_trade_state_sources(lake: DataLake) -> None:
    missing = [
        dataset
        for dataset in _REQUIRED_TRADE_STATE_DATASETS.values()
        if not lake.dataset_path("raw", dataset).exists()
    ]
    if missing:
        raise BacktestDataIntegrityError(
            code="TRADE_STATE_SOURCE_NOT_READY",
            message="required trade-state source datasets are unavailable",
            field="trade_state",
            details={"missing_datasets": missing},
        )


def column_quality(frame: pd.DataFrame, column: str) -> dict[str, object]:
    quality = frame.attrs.get("column_quality")
    if isinstance(quality, dict) and isinstance(quality.get(column), dict):
        return dict(quality[column])
    if column not in frame.columns:
        return {
            "source": "missing_column",
            "imputed": False,
            "usable_for_factor": False,
        }
    return {"source": "unknown", "imputed": False, "usable_for_factor": None}


def is_column_usable_for_factor(frame: pd.DataFrame, column: str) -> bool:
    quality = column_quality(frame, column)
    return quality.get("usable_for_factor") is not False


def enrich_trade_states(
    bars: pd.DataFrame,
    *,
    suspend: pd.DataFrame | None = None,
    stk_limit: pd.DataFrame | None = None,
    namechange: pd.DataFrame | None = None,
    stock_basic: pd.DataFrame | None = None,
) -> pd.DataFrame:
    if suspend is None or stk_limit is None or namechange is None:
        raise BacktestDataIntegrityError(
            code="TRADE_STATE_SOURCE_NOT_READY",
            message="opening trade-state normalization requires every stock source",
            field="trade_state",
        )
    return normalize_stock_opening_trade_state(
        bars,
        suspend=suspend,
        stk_limit=stk_limit,
        namechange=namechange,
    )


def _merge_column_quality(frames: list[pd.DataFrame]) -> dict[str, dict[str, object]]:
    merged: dict[str, dict[str, object]] = {}
    for frame in frames:
        quality = frame.attrs.get("column_quality")
        if not isinstance(quality, dict):
            continue
        for column, entry in quality.items():
            if not isinstance(entry, dict):
                continue
            existing = merged.get(str(column))
            if existing is None:
                merged[str(column)] = dict(entry)
                continue
            if (
                existing.get("usable_for_factor") is False
                or entry.get("usable_for_factor") is False
            ):
                existing["usable_for_factor"] = False
            if existing.get("source") != entry.get("source"):
                existing["source"] = "mixed"
            existing["imputed"] = bool(existing.get("imputed")) or bool(entry.get("imputed"))
    return merged


def _parse_date(value: str | date) -> date:
    if isinstance(value, date):
        return value
    for fmt in ("%Y%m%d", "%Y-%m-%d"):
        try:
            return datetime.strptime(value, fmt).date()
        except ValueError:
            continue
    return datetime.fromisoformat(value).date()


def _read_optional_dataset(lake: DataLake, layer: str, name: str) -> pd.DataFrame:
    if not lake.dataset_path(layer, name).exists():
        return pd.DataFrame()
    return lake.read_parquet(layer, name)


def _filter_namechange_overlap(
    namechange: pd.DataFrame,
    *,
    start: str | date | None,
    end: str | date | None,
) -> pd.DataFrame:
    if namechange.empty or not {"start_date", "end_date"}.intersection(namechange.columns):
        return namechange
    data = namechange.copy()
    requested_start = _parse_date(start) if start is not None else date(1900, 1, 1)
    requested_end = _parse_date(end) if end is not None else date(2099, 12, 31)
    if "start_date" in data.columns:
        period_start = _coerce_date_series(data["start_date"], default="19000101")
    else:
        period_start = pd.Series([date(1900, 1, 1)] * len(data), index=data.index)
    if "end_date" in data.columns:
        period_end = _coerce_date_series(data["end_date"], default="20991231")
    else:
        period_end = pd.Series([date(2099, 12, 31)] * len(data), index=data.index)
    return data[(period_start <= requested_end) & (period_end >= requested_start)]


def _coerce_date_series(values: pd.Series, *, default: str | None = None) -> pd.Series:
    text = values.astype(str).str.strip()
    if default is not None:
        text = text.replace(
            {
                "": default,
                "NaT": default,
                "None": default,
                "nan": default,
                "<NA>": default,
            }
        )
        text = text.where(values.notna(), default)
    parsed = pd.to_datetime(text, format="%Y%m%d", errors="coerce")
    missing = parsed.isna()
    if missing.any():
        parsed.loc[missing] = pd.to_datetime(text[missing], errors="coerce")
    return parsed.dt.date
