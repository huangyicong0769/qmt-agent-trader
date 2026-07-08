"""Research-facing projections built from raw and silver lake datasets."""

from __future__ import annotations

from datetime import date, datetime

import pandas as pd

from qmt_agent_trader.data.bars import load_daily_bars
from qmt_agent_trader.data.fundamentals import load_financials_asof
from qmt_agent_trader.data.storage import DataLake


def load_daily_market(
    lake: DataLake,
    *,
    symbols: list[str] | None = None,
    start: str | date | None = None,
    end: str | date | None = None,
    fields: list[str] | None = None,
) -> pd.DataFrame:
    """Load canonical daily market rows from silver when present, else raw bars."""

    if lake.dataset_path("silver", "daily_market").exists():
        frame = lake.read_parquet_filtered(
            "silver",
            "daily_market",
            start=start,
            end=end,
            symbols=symbols,
        )
        frame = frame.rename(columns={"ts_code": "symbol", "vol": "volume"})
    else:
        frame = load_daily_bars(lake, start=start, end=end, symbols=symbols)
    return _project_columns(frame, fields, required=["symbol", "trade_date"])


def load_financial_snapshot(
    lake: DataLake,
    *,
    as_of_date: str | date,
    symbols: list[str] | None = None,
    fields: list[str] | None = None,
) -> pd.DataFrame:
    """Load the latest point-in-time financial snapshot visible by ``as_of_date``."""

    as_of = _parse_date(as_of_date)
    if lake.dataset_path("silver", "financial_reports_wide").exists():
        reports = lake.read_parquet("silver", "financial_reports_wide")
        frame = _latest_financial_reports(reports, as_of=as_of, symbols=symbols)
    else:
        frame = load_financials_asof(
            lake,
            as_of_date=as_of,
            symbols=symbols,
            fields=fields,
        )
    return _project_columns(frame, fields, required=["symbol", "as_of_date"])


def load_macro_snapshot(
    lake: DataLake,
    *,
    as_of_date: str | date,
    macro_ids: list[str] | None = None,
) -> pd.DataFrame:
    """Return macro observations visible by ``as_of_date`` in long form."""

    as_of = _parse_date(as_of_date)
    if lake.dataset_path("silver", "macro_series").exists():
        frame = lake.read_parquet("silver", "macro_series")
    else:
        frame = _build_macro_long_from_raw(lake)
    if frame.empty:
        return frame
    data = frame.copy()
    if "visible_date" in data.columns:
        data["visible_date"] = _coerce_timestamp(data["visible_date"])
        data = data[data["visible_date"] <= pd.Timestamp(as_of)]
    if macro_ids and "macro_id" in data.columns:
        data = data[data["macro_id"].astype(str).isin(macro_ids)]
    sort_columns = [column for column in ("macro_id", "period") if column in data.columns]
    return data.sort_values(sort_columns).reset_index(drop=True)


def load_corporate_actions(
    lake: DataLake,
    *,
    symbols: list[str] | None = None,
    as_of_date: str | date | None = None,
) -> pd.DataFrame:
    """Load corporate action events from silver or raw dividend/namechange datasets."""

    if lake.dataset_path("silver", "corporate_actions").exists():
        frame = lake.read_parquet("silver", "corporate_actions")
    else:
        frame = _build_corporate_actions_from_raw(lake)
    if frame.empty:
        return frame
    data = frame.rename(columns={"ts_code": "symbol"}).copy()
    if symbols and "symbol" in data.columns:
        data = data[data["symbol"].astype(str).isin(symbols)]
    if as_of_date is not None and "visible_date" in data.columns:
        as_of = _parse_date(as_of_date)
        data["visible_date"] = _coerce_timestamp(data["visible_date"])
        data = data[data["visible_date"] <= pd.Timestamp(as_of)]
    return data.reset_index(drop=True)


def build_research_feature_frame(
    lake: DataLake,
    *,
    symbols: list[str],
    start: str | date,
    end: str | date,
    include_financials: bool = True,
    include_macro: bool = False,
    include_corporate_actions: bool = False,
) -> pd.DataFrame:
    """Build a PIT-safe daily feature frame for research workflows."""

    market = load_daily_market(lake, symbols=symbols, start=start, end=end)
    if market.empty:
        return market
    frame = market.copy()
    frame["trade_date"] = _coerce_timestamp(frame["trade_date"])
    if include_financials:
        financials = _financial_history_for_feature_join(lake, symbols=symbols)
        frame = _join_latest_by_visible_date(frame, financials)
    if include_macro:
        macro = load_macro_snapshot(lake, as_of_date=end)
        frame.attrs["macro_snapshot_rows"] = len(macro)
    if include_corporate_actions:
        actions = load_corporate_actions(lake, symbols=symbols, as_of_date=end)
        if not actions.empty and "symbol" in actions.columns:
            action_counts = (
                actions.groupby("symbol", as_index=False)
                .size()
                .rename(columns={"size": "corporate_action_count"})
            )
            frame = frame.merge(action_counts, on="symbol", how="left")
            frame["corporate_action_count"] = frame["corporate_action_count"].fillna(0).astype(int)
    return frame.sort_values(["symbol", "trade_date"]).reset_index(drop=True)


def _latest_financial_reports(
    reports: pd.DataFrame,
    *,
    as_of: date,
    symbols: list[str] | None,
) -> pd.DataFrame:
    data = reports.rename(columns={"ts_code": "symbol"}).copy()
    if data.empty or "symbol" not in data.columns or "visible_date" not in data.columns:
        return pd.DataFrame()
    data["visible_date"] = _coerce_timestamp(data["visible_date"])
    data = data[data["visible_date"] <= pd.Timestamp(as_of)]
    if symbols:
        data = data[data["symbol"].astype(str).isin(symbols)]
    if data.empty:
        return pd.DataFrame()
    if "report_period" in data.columns:
        data["report_period"] = _coerce_timestamp(data["report_period"])
    sort_columns = [
        column
        for column in ("symbol", "visible_date", "report_period")
        if column in data.columns
    ]
    latest = data.sort_values(sort_columns).groupby("symbol", as_index=False).tail(1).copy()
    latest["as_of_date"] = as_of
    return latest.reset_index(drop=True)


def _financial_history_for_feature_join(lake: DataLake, *, symbols: list[str]) -> pd.DataFrame:
    if lake.dataset_path("silver", "financial_reports_wide").exists():
        frame = lake.read_parquet("silver", "financial_reports_wide").rename(
            columns={"ts_code": "symbol"}
        )
    else:
        frames: list[pd.DataFrame] = []
        for source in (
            "tushare/fina_indicator",
            "tushare/income",
            "tushare/balancesheet",
            "tushare/cashflow",
        ):
            if not lake.dataset_path("raw", source).exists():
                continue
            raw = lake.read_parquet("raw", source)
            raw = raw.rename(columns={"ts_code": "symbol"}).copy()
            raw["source"] = source
            raw["visible_date"] = raw.get("f_ann_date", raw.get("ann_date"))
            raw["report_period"] = raw.get("end_date")
            frames.append(raw)
        frame = pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()
    if frame.empty or "symbol" not in frame.columns or "visible_date" not in frame.columns:
        return pd.DataFrame()
    frame = frame[frame["symbol"].astype(str).isin(symbols)].copy()
    frame["visible_date"] = _coerce_timestamp(frame["visible_date"])
    if "report_period" in frame.columns:
        frame["report_period"] = _coerce_timestamp(frame["report_period"])
    return frame.dropna(subset=["visible_date"]).sort_values(["symbol", "visible_date"])


def _join_latest_by_visible_date(market: pd.DataFrame, financials: pd.DataFrame) -> pd.DataFrame:
    if financials.empty:
        return market
    joined: list[pd.DataFrame] = []
    reserved = {"symbol", "trade_date"}
    drop_columns = [column for column in ("ts_code", "updated_at") if column in financials.columns]
    right = financials.drop(columns=drop_columns).copy()
    for symbol, left_group in market.groupby("symbol", sort=False):
        right_group = right[right["symbol"].astype(str) == str(symbol)].copy()
        if right_group.empty:
            joined.append(left_group)
            continue
        merged = pd.merge_asof(
            left_group.sort_values("trade_date"),
            right_group.sort_values("visible_date"),
            by="symbol",
            left_on="trade_date",
            right_on="visible_date",
            direction="backward",
            suffixes=("", "_financial"),
        )
        duplicate_columns = [
            column
            for column in merged.columns
            if column.endswith("_financial") and column.removesuffix("_financial") in reserved
        ]
        if duplicate_columns:
            merged = merged.drop(columns=duplicate_columns)
        joined.append(merged)
    return pd.concat(joined, ignore_index=True, sort=False)


def _build_macro_long_from_raw(lake: DataLake) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for source, period_column, period_type in (
        ("tushare/cn_gdp", "quarter", "quarterly"),
        ("tushare/cn_cpi", "month", "monthly"),
        ("tushare/cn_ppi", "month", "monthly"),
        ("tushare/shibor", "date", "daily"),
    ):
        if not lake.dataset_path("raw", source).exists():
            continue
        raw = lake.read_parquet("raw", source)
        if raw.empty or period_column not in raw.columns:
            continue
        value_columns = [column for column in raw.columns if column != period_column]
        melted = raw.melt(
            id_vars=[period_column],
            value_vars=value_columns,
            var_name="source_field",
            value_name="value",
        )
        melted["macro_id"] = source.replace("tushare/", "")
        melted["macro_id"] = melted["macro_id"] + "." + melted["source_field"].astype(str)
        melted["period"] = melted[period_column].astype(str)
        melted["period_type"] = period_type
        melted["visible_date"] = _macro_visible_date(melted["period"], period_type=period_type)
        melted["source_api"] = source
        frames.append(
            melted[
                [
                    "macro_id",
                    "period",
                    "period_type",
                    "value",
                    "visible_date",
                    "source_api",
                    "source_field",
                ]
            ]
        )
    return pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()


def _build_corporate_actions_from_raw(lake: DataLake) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    dividend = (
        lake.read_parquet("raw", "tushare/dividend")
        if lake.dataset_path("raw", "tushare/dividend").exists()
        else pd.DataFrame()
    )
    if not dividend.empty:
        data = dividend.copy()
        data["action_type"] = "dividend"
        data["visible_date"] = data.get("ann_date")
        frames.append(data)
    namechange = (
        lake.read_parquet("raw", "tushare/namechange")
        if lake.dataset_path("raw", "tushare/namechange").exists()
        else pd.DataFrame()
    )
    if not namechange.empty:
        data = namechange.copy()
        data["action_type"] = "namechange"
        data["visible_date"] = data.get("start_date")
        frames.append(data)
    return pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()


def _project_columns(
    frame: pd.DataFrame,
    fields: list[str] | None,
    *,
    required: list[str],
) -> pd.DataFrame:
    if frame.empty or fields is None:
        return frame.reset_index(drop=True)
    columns = [column for column in [*required, *fields] if column in frame.columns]
    return frame.loc[:, list(dict.fromkeys(columns))].reset_index(drop=True)


def _parse_date(value: str | date) -> date:
    if isinstance(value, date):
        return value
    for fmt in ("%Y%m%d", "%Y-%m-%d"):
        try:
            return datetime.strptime(value, fmt).date()
        except ValueError:
            continue
    return datetime.fromisoformat(value).date()


def _coerce_date(values: pd.Series) -> pd.Series:
    return _coerce_timestamp(values).dt.date


def _coerce_timestamp(values: pd.Series) -> pd.Series:
    text = values.astype(str).str.strip()
    text = text.replace({"": None, "NaT": None, "None": None, "nan": None, "<NA>": None})
    parsed = pd.to_datetime(text, format="%Y%m%d", errors="coerce")
    missing = parsed.isna()
    if missing.any():
        parsed.loc[missing] = pd.to_datetime(text[missing], format="%Y%m", errors="coerce")
        parsed.loc[missing] = parsed.loc[missing] + pd.offsets.MonthEnd(0)
    missing = parsed.isna()
    if missing.any():
        parsed.loc[missing] = pd.to_datetime(text[missing], errors="coerce")
    return parsed.astype("datetime64[ns]")


def _macro_visible_date(periods: pd.Series, *, period_type: str) -> pd.Series:
    text = periods.astype(str).str.strip()
    if period_type == "monthly":
        parsed = pd.to_datetime(text, format="%Y%m", errors="coerce")
        return (parsed + pd.offsets.MonthEnd(0)).astype("datetime64[ns]")
    if period_type == "quarterly":
        parsed = (
            pd.PeriodIndex(text, freq="Q")
            .to_timestamp(how="end")
            .to_series(index=periods.index)
        )
        return parsed.dt.normalize().astype("datetime64[ns]")
    return _coerce_timestamp(periods)
