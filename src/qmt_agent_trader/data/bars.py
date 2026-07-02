"""Canonical bar readers backed by the data lake."""

from __future__ import annotations

from datetime import date, datetime

import pandas as pd

from qmt_agent_trader.data.storage import DataLake

CANONICAL_BAR_COLUMNS = [
    "symbol",
    "trade_date",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "amount",
    "turnover",
    "suspended",
    "limit_up",
    "limit_down",
    "st",
]


def normalize_tushare_daily(frame: pd.DataFrame) -> pd.DataFrame:
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

    rename_map = {"ts_code": "symbol", "vol": "volume"}
    data = data.rename(columns=rename_map)
    required = {"symbol", "trade_date", "open", "high", "low", "close"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"tushare daily bars missing columns: {sorted(missing)}")

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
    for column in ["suspended", "limit_up", "limit_down", "st"]:
        if column not in data.columns:
            data[column] = False

    normalized = (
        data[CANONICAL_BAR_COLUMNS]
        .drop_duplicates(["symbol", "trade_date"], keep="last")
        .sort_values(["symbol", "trade_date"])
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
    raw_frames = [
        lake.read_parquet_filtered(
            "raw",
            "tushare/daily",
            columns=daily_columns,
            start=start,
            end=end,
            symbols=symbols,
        ),
        lake.read_parquet_filtered(
            "raw",
            "tushare/fund_daily",
            columns=daily_columns,
            start=start,
            end=end,
            symbols=symbols,
        ),
    ]
    normalized = [normalize_tushare_daily(frame) for frame in raw_frames if not frame.empty]
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
    if bars.empty:
        return bars.reset_index(drop=True)

    bars = enrich_trade_states(
        bars,
        suspend=lake.read_parquet_filtered(
            "raw",
            "tushare/suspend_d",
            columns=["ts_code", "trade_date", "suspend_type"],
            start=start,
            end=end,
            symbols=symbols,
        ),
        stk_limit=lake.read_parquet_filtered(
            "raw",
            "tushare/stk_limit",
            columns=["ts_code", "trade_date", "up_limit", "down_limit"],
            start=start,
            end=end,
            symbols=symbols,
        ),
        namechange=_filter_namechange_overlap(
            lake.read_parquet_filtered(
                "raw",
                "tushare/namechange",
                columns=["ts_code", "name", "start_date", "end_date"],
                symbols=symbols,
            ),
            start=start,
            end=end,
        ),
        stock_basic=lake.read_parquet_filtered(
            "raw",
            "tushare/stock_basic",
            columns=["ts_code", "name"],
            symbols=symbols,
        ),
    )

    result = bars.sort_values(["symbol", "trade_date"]).reset_index(drop=True)
    result.attrs["column_quality"] = bars.attrs.get("column_quality", {})
    return result


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
    enriched = bars.copy()
    if suspend is not None and not suspend.empty:
        enriched = _apply_suspend_flags(enriched, suspend)
    if stk_limit is not None and not stk_limit.empty:
        enriched = _apply_limit_flags(enriched, stk_limit)
    if stock_basic is not None and not stock_basic.empty:
        enriched = _apply_current_st_flags(enriched, stock_basic)
    if namechange is not None and not namechange.empty:
        enriched = _apply_historical_st_flags(enriched, namechange)
    for column in ["suspended", "limit_up", "limit_down", "st"]:
        enriched[column] = enriched[column].fillna(False).astype(bool)
    enriched.attrs["column_quality"] = bars.attrs.get("column_quality", {})
    return enriched


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


def _state_key_frame(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.copy()
    data = data.rename(columns={"ts_code": "symbol"})
    if "trade_date" in data.columns:
        data["trade_date"] = _coerce_date_series(data["trade_date"])
    return data


def _apply_suspend_flags(bars: pd.DataFrame, suspend: pd.DataFrame) -> pd.DataFrame:
    data = _state_key_frame(suspend)
    if not {"symbol", "trade_date"}.issubset(data.columns):
        return bars
    keys = data[["symbol", "trade_date"]].drop_duplicates()
    keys["suspended_state"] = True
    merged = bars.merge(keys, on=["symbol", "trade_date"], how="left")
    merged["suspended"] = merged["suspended"] | merged["suspended_state"].fillna(False)
    return merged.drop(columns=["suspended_state"])


def _apply_limit_flags(bars: pd.DataFrame, stk_limit: pd.DataFrame) -> pd.DataFrame:
    data = _state_key_frame(stk_limit)
    required = {"symbol", "trade_date", "up_limit", "down_limit"}
    if not required.issubset(data.columns):
        return bars
    merged = bars.merge(
        data[["symbol", "trade_date", "up_limit", "down_limit"]],
        on=["symbol", "trade_date"],
        how="left",
    )
    tolerance = 1e-6
    at_up_open = merged["up_limit"].notna() & (merged["open"] >= merged["up_limit"] - tolerance)
    at_up_close = merged["up_limit"].notna() & (merged["close"] >= merged["up_limit"] - tolerance)
    at_down_open = merged["down_limit"].notna() & (
        merged["open"] <= merged["down_limit"] + tolerance
    )
    at_down_close = merged["down_limit"].notna() & (
        merged["close"] <= merged["down_limit"] + tolerance
    )
    merged["limit_up"] = merged["limit_up"] | at_up_open | at_up_close
    merged["limit_down"] = merged["limit_down"] | at_down_open | at_down_close
    return merged.drop(columns=["up_limit", "down_limit"])


def _apply_current_st_flags(bars: pd.DataFrame, stock_basic: pd.DataFrame) -> pd.DataFrame:
    data = stock_basic.rename(columns={"ts_code": "symbol"}).copy()
    if not {"symbol", "name"}.issubset(data.columns):
        return bars
    st_mask = data["name"].astype(str).str.contains("ST", case=False, na=False)
    st_symbols = set(data.loc[st_mask, "symbol"])
    if not st_symbols:
        return bars
    enriched = bars.copy()
    enriched["st"] = enriched["st"] | enriched["symbol"].isin(st_symbols)
    return enriched


def _apply_historical_st_flags(bars: pd.DataFrame, namechange: pd.DataFrame) -> pd.DataFrame:
    data = namechange.rename(columns={"ts_code": "symbol"}).copy()
    if not {"symbol", "name", "start_date"}.issubset(data.columns):
        return bars
    data["start_date"] = _coerce_date_series(data["start_date"])
    if "end_date" in data.columns:
        data["end_date"] = _coerce_date_series(data["end_date"], default="20991231")
    else:
        data["end_date"] = date(2099, 12, 31)
    st_periods = data[data["name"].astype(str).str.contains("ST", case=False, na=False)]
    if st_periods.empty:
        return bars
    enriched = bars.copy()
    symbol_indices = enriched.groupby("symbol", sort=False).indices
    for symbol, periods in st_periods.groupby("symbol", sort=False):
        index = symbol_indices.get(symbol)
        if index is None or len(index) == 0:
            continue
        dates = enriched.iloc[index]["trade_date"]
        mask = pd.Series(False, index=dates.index)
        for row in periods.itertuples(index=False):
            mask |= (dates >= row.start_date) & (dates <= row.end_date)
        if mask.any():
            enriched.loc[mask[mask].index, "st"] = True
    return enriched


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
