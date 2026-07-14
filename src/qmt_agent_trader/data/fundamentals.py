"""Point-in-time fundamental data readers."""

from __future__ import annotations

from datetime import date, datetime
from typing import Any

import pandas as pd

from qmt_agent_trader.data.integrity import require_unique_symbol_dates
from qmt_agent_trader.data.storage import DataLake

DAILY_BASIC_DATASET = "tushare/daily_basic"
FINANCIAL_DATASETS = {
    "income": "tushare/income",
    "balancesheet": "tushare/balancesheet",
    "cashflow": "tushare/cashflow",
    "fina_indicator": "tushare/fina_indicator",
    "dividend": "tushare/dividend",
}

DEFAULT_FUNDAMENTAL_FIELDS = [
    "pe_ttm",
    "pb",
    "dv_ttm",
    "total_mv",
    "circ_mv",
    "roe",
    "roe_dt",
    "roa",
    "gross_margin",
    "debt_to_assets",
    "current_ratio",
    "net_profit_yoy",
    "revenue_yoy",
]

DAILY_BASIC_FIELDS = [
    "pe",
    "pe_ttm",
    "pb",
    "ps",
    "ps_ttm",
    "dv_ratio",
    "dv_ttm",
    "total_mv",
    "circ_mv",
    "turnover_rate",
    "volume_ratio",
]

FINANCIAL_IDENTITY_FIELDS = [
    "symbol",
    "period_end",
    "announced_at",
    "actual_announced_at",
    "visible_date",
    "report_type",
    "statement_type",
    "update_flag",
    "source",
    "pit_safe",
]


def load_daily_basic_snapshot(
    lake: DataLake,
    *,
    as_of_date: str | date,
    symbols: list[str] | None = None,
    fields: list[str] | None = None,
) -> pd.DataFrame:
    if not lake.dataset_path("raw", DAILY_BASIC_DATASET).exists():
        return pd.DataFrame()
    frame = lake.read_parquet("raw", DAILY_BASIC_DATASET)
    if frame.empty or "trade_date" not in frame.columns or "ts_code" not in frame.columns:
        return pd.DataFrame()

    as_of = parse_date(as_of_date)
    data = frame.rename(columns={"ts_code": "symbol"}).copy()
    data["trade_date"] = _coerce_date(data["trade_date"])
    data = data[data["trade_date"] <= as_of]
    if symbols:
        data = data[data["symbol"].astype(str).isin(symbols)]
    if data.empty:
        return pd.DataFrame()

    keep_fields = fields or DAILY_BASIC_FIELDS
    columns = ["symbol", "trade_date", *[field for field in keep_fields if field in data.columns]]
    require_unique_symbol_dates(
        data,
        symbol_column="symbol",
        date_column="trade_date",
        code="DUPLICATE_EXACT_FACTOR_INPUT",
        field=f"raw/{DAILY_BASIC_DATASET}",
    )
    latest = (
        data[columns]
        .sort_values(["symbol", "trade_date"])
        .groupby("symbol", as_index=False)
        .tail(1)
        .reset_index(drop=True)
    )
    latest["as_of_date"] = as_of
    latest["source"] = DAILY_BASIC_DATASET
    latest["data_status"] = "OK"
    return latest


def load_financials_asof(
    lake: DataLake,
    *,
    as_of_date: str | date,
    symbols: list[str] | None = None,
    fields: list[str] | None = None,
) -> pd.DataFrame:
    as_of = parse_date(as_of_date)
    frames: list[pd.DataFrame] = []
    for statement_type, dataset in FINANCIAL_DATASETS.items():
        if not lake.dataset_path("raw", dataset).exists():
            continue
        raw = lake.read_parquet("raw", dataset)
        normalized = normalize_financial_statement(
            raw,
            statement_type=statement_type,
            source=dataset,
        )
        if not normalized.empty:
            frames.append(normalized)
    if not frames:
        return pd.DataFrame()

    data = pd.concat(frames, ignore_index=True)
    for column in ("visible_date", "period_end", "announced_at", "actual_announced_at"):
        if column in data.columns:
            data[column] = pd.to_datetime(data[column], errors="coerce").dt.date
    data = data[data["pit_safe"]]
    data = data[data["visible_date"] <= as_of]
    if symbols:
        data = data[data["symbol"].astype(str).isin(symbols)]
    if data.empty:
        return pd.DataFrame()

    requested_fields = fields or [
        field for field in DEFAULT_FUNDAMENTAL_FIELDS if field not in DAILY_BASIC_FIELDS
    ]
    value_columns = [field for field in requested_fields if field in data.columns]
    if not value_columns:
        return _latest_financial_metadata(data, as_of)

    latest_values: list[pd.DataFrame] = []
    for field in value_columns:
        field_data = data[[*FINANCIAL_IDENTITY_FIELDS, field]].dropna(subset=[field])
        if field_data.empty:
            continue
        latest = (
            field_data.sort_values(
                ["symbol", "visible_date", "period_end", "update_flag"],
                na_position="first",
            )
            .groupby("symbol", as_index=False)
            .tail(1)
            .loc[:, ["symbol", field]]
        )
        latest_values.append(latest)
    metadata = _latest_financial_metadata(data, as_of)
    if not latest_values:
        return metadata
    wide = latest_values[0]
    for frame in latest_values[1:]:
        wide = wide.merge(frame, on="symbol", how="outer")
    return metadata.merge(wide, on="symbol", how="outer")


def load_fundamentals_asof(
    lake: DataLake,
    *,
    as_of_date: str | date,
    symbols: list[str] | None = None,
    fields: list[str] | None = None,
    include_daily_basic: bool = True,
    include_financials: bool = True,
) -> pd.DataFrame:
    as_of = parse_date(as_of_date)
    requested_fields = fields or DEFAULT_FUNDAMENTAL_FIELDS
    frames: list[pd.DataFrame] = []
    if include_daily_basic:
        daily_fields = [field for field in requested_fields if field in DAILY_BASIC_FIELDS]
        daily = load_daily_basic_snapshot(
            lake,
            as_of_date=as_of,
            symbols=symbols,
            fields=daily_fields or DAILY_BASIC_FIELDS,
        )
        if not daily.empty:
            frames.append(daily.drop(columns=["source"], errors="ignore"))
    if include_financials:
        financial_fields = [field for field in requested_fields if field not in DAILY_BASIC_FIELDS]
        financial = load_financials_asof(
            lake,
            as_of_date=as_of,
            symbols=symbols,
            fields=financial_fields or None,
        )
        if not financial.empty:
            frames.append(financial)
    if not frames:
        return pd.DataFrame()

    result = frames[0]
    for frame in frames[1:]:
        result = result.merge(frame, on="symbol", how="outer", suffixes=("", "_financial"))
    result = result.drop(
        columns=["as_of_date_financial", "data_status_financial"],
        errors="ignore",
    )
    result["as_of_date"] = as_of
    if "data_status" not in result.columns:
        result["data_status"] = "OK"
    if symbols:
        result = result[result["symbol"].isin(symbols)]
    return result.sort_values("symbol").reset_index(drop=True)


def normalize_financial_statement(
    frame: pd.DataFrame,
    *,
    statement_type: str,
    source: str,
) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    data = frame.rename(columns={"ts_code": "symbol"}).copy()
    if "_empty" in data.columns:
        data = data.drop(columns=["_empty"])
    if data.empty or "symbol" not in data.columns:
        return pd.DataFrame()
    data["period_end"] = (
        _coerce_date(data["end_date"]) if "end_date" in data.columns else pd.NaT
    )
    ann = (
        _coerce_date(data["ann_date"])
        if "ann_date" in data.columns
        else pd.Series(pd.NaT, index=data.index)
    )
    f_ann = (
        _coerce_date(data["f_ann_date"])
        if "f_ann_date" in data.columns
        else pd.Series(pd.NaT, index=data.index)
    )
    data["actual_announced_at"] = f_ann
    data["announced_at"] = f_ann.where(f_ann.notna(), ann)
    data["visible_date"] = data["announced_at"]
    data["pit_safe"] = data["visible_date"].notna()
    data["statement_type"] = statement_type
    data["source"] = source
    if "report_type" not in data.columns:
        data["report_type"] = None
    if "update_flag" not in data.columns:
        data["update_flag"] = None
    return data


def _latest_financial_metadata(data: pd.DataFrame, as_of: date) -> pd.DataFrame:
    latest = (
        data.sort_values(["symbol", "visible_date", "period_end"], na_position="first")
        .groupby("symbol", as_index=False)
        .tail(1)
        .loc[:, ["symbol", "period_end", "announced_at"]]
        .rename(
            columns={
                "period_end": "latest_period_end",
                "announced_at": "latest_announced_at",
            }
        )
        .reset_index(drop=True)
    )
    latest["as_of_date"] = as_of
    latest["data_status"] = "OK"
    return latest


def parse_date(value: str | date) -> date:
    if isinstance(value, date):
        return value
    for fmt in ("%Y%m%d", "%Y-%m-%d"):
        try:
            return datetime.strptime(value, fmt).date()
        except ValueError:
            continue
    return datetime.fromisoformat(value).date()


def _coerce_date(values: pd.Series) -> pd.Series:
    text = values.astype(str).str.strip()
    text = text.replace({"": None, "NaT": None, "None": None, "nan": None, "<NA>": None})
    parsed = pd.to_datetime(text, format="%Y%m%d", errors="coerce")
    missing = parsed.isna()
    if missing.any():
        parsed.loc[missing] = pd.to_datetime(text[missing], errors="coerce")
    return parsed.dt.date


def records_jsonable(frame: pd.DataFrame) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for record in frame.to_dict(orient="records"):
        records.append(
            {
                key: (f"{value:%Y-%m-%d}" if isinstance(value, date) else value)
                for key, value in record.items()
                if not pd.isna(value)
            }
        )
    return records
