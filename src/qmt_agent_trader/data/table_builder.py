"""Build silver tables from registry-driven raw Tushare datasets."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from typing import Any

import pandas as pd

from qmt_agent_trader.data.integrity import require_unique_keys
from qmt_agent_trader.data.storage import DataLake

ALLOWED_SILVER_TABLES = {
    "security_master",
    "trade_calendar",
    "daily_market",
    "index_daily",
    "financial_reports_wide",
    "financial_current_wide",
    "corporate_actions",
    "macro_series",
}


class DataTableBuilder:
    def __init__(self, lake: DataLake) -> None:
        self.lake = lake

    def build(self, table: str, *, snapshot_as_of_date: str | None = None) -> dict[str, Any]:
        if table not in ALLOWED_SILVER_TABLES:
            return {
                "status": "INVALID_REQUEST",
                "message": f"unknown or disallowed silver table: {table}",
                "allowed_tables": sorted(ALLOWED_SILVER_TABLES),
            }
        frame = getattr(self, f"_build_{table}")(snapshot_as_of_date=snapshot_as_of_date)
        if not frame.empty:
            frame = frame.copy()
            frame["updated_at"] = datetime.now(tz=UTC).isoformat()
        key_columns = _silver_keys(table)
        path = self.lake.write_incremental_parquet(
            frame,
            "silver",
            table,
            key_columns=key_columns,
        )
        self.lake.register_parquet(table, "silver", table)
        return {"status": "built", "table": table, "rows": len(frame), "path": str(path)}

    def _build_security_master(self, *, snapshot_as_of_date: str | None) -> pd.DataFrame:
        _ = snapshot_as_of_date
        frames: list[pd.DataFrame] = []
        stock = _read(self.lake, "tushare/stock_basic")
        if not stock.empty:
            stock = _select(
                stock,
                {
                    "ts_code": "ts_code",
                    "symbol": "symbol",
                    "name": "name",
                    "exchange": "exchange",
                    "market": "market",
                    "industry": "industry",
                    "area": "area",
                    "list_date": "list_date",
                    "delist_date": "delist_date",
                },
            )
            stock["asset_type"] = "stock"
            stock["is_active"] = stock.get("delist_date", pd.Series(dtype=object)).isna()
            stock["source"] = "tushare.stock_basic"
            frames.append(stock)
        fund = _read(self.lake, "tushare/fund_basic")
        if not fund.empty:
            fund = _select(
                fund,
                {
                    "ts_code": "ts_code",
                    "name": "name",
                    "market": "market",
                    "list_date": "list_date",
                    "delist_date": "delist_date",
                },
            )
            fund["asset_type"] = "fund"
            fund["symbol"] = fund["ts_code"].astype(str).str.split(".").str[0]
            fund["exchange"] = fund["ts_code"].astype(str).str.split(".").str[-1]
            fund["industry"] = None
            fund["area"] = None
            fund["is_active"] = fund.get("delist_date", pd.Series(dtype=object)).isna()
            fund["source"] = "tushare.fund_basic"
            frames.append(fund)
        index = _read(self.lake, "tushare/index_basic")
        if not index.empty:
            index = _select(
                index,
                {
                    "ts_code": "ts_code",
                    "name": "name",
                    "market": "market",
                    "list_date": "list_date",
                    "exp_date": "delist_date",
                },
            )
            index["asset_type"] = "index"
            index["symbol"] = index["ts_code"].astype(str).str.split(".").str[0]
            index["exchange"] = index["market"]
            index["industry"] = None
            index["area"] = None
            index["is_active"] = index.get("delist_date", pd.Series(dtype=object)).isna()
            index["source"] = "tushare.index_basic"
            frames.append(index)
        return _concat(frames, ["ts_code"])

    def _build_trade_calendar(self, *, snapshot_as_of_date: str | None) -> pd.DataFrame:
        _ = snapshot_as_of_date
        frame = _read(self.lake, "tushare/trade_cal")
        if frame.empty:
            return pd.DataFrame(
                columns=["exchange", "cal_date", "is_open", "pretrade_date", "source"]
            )
        frame = _select(
            frame,
            {
                "exchange": "exchange",
                "cal_date": "cal_date",
                "is_open": "is_open",
                "pretrade_date": "pretrade_date",
            },
        )
        frame["source"] = "tushare.trade_cal"
        return frame

    def _build_daily_market(self, *, snapshot_as_of_date: str | None) -> pd.DataFrame:
        _ = snapshot_as_of_date
        frames: list[pd.DataFrame] = []
        for name, asset_type in (("tushare/daily", "stock"), ("tushare/fund_daily", "fund")):
            frame = _read(self.lake, name)
            if not frame.empty:
                frame = frame.copy()
                frame["asset_type"] = asset_type
                frame["source_flags"] = name
                frames.append(frame)
        base = _concat(frames, ["ts_code", "trade_date"])
        for name in ("tushare/daily_basic", "tushare/suspend_d", "tushare/stk_limit"):
            addon = _read(self.lake, name)
            if not addon.empty and not base.empty:
                base = base.merge(
                    addon,
                    on=["ts_code", "trade_date"],
                    how="left",
                    suffixes=("", "_dup"),
                )
                base = base[[column for column in base.columns if not column.endswith("_dup")]]
        if not base.empty:
            base["source"] = "tushare"
        return base

    def _build_index_daily(self, *, snapshot_as_of_date: str | None) -> pd.DataFrame:
        _ = snapshot_as_of_date
        frame = _read(self.lake, "tushare/index_daily")
        if frame.empty:
            return pd.DataFrame(columns=["ts_code", "trade_date"])
        frame = frame.copy()
        frame["source"] = "tushare.index_daily"
        return frame

    def _build_financial_reports_wide(self, *, snapshot_as_of_date: str | None) -> pd.DataFrame:
        _ = snapshot_as_of_date
        merged: pd.DataFrame | None = None
        source_flags: dict[tuple[str, str], set[str]] = {}
        for name in ("income", "balancesheet", "cashflow", "fina_indicator"):
            frame = _read(self.lake, f"tushare/{name}")
            if frame.empty:
                continue
            normalized = _normalize_financial_report_frame(frame, source_name=name)
            if normalized.empty:
                continue
            for row in normalized[["ts_code", "report_period"]].itertuples(index=False):
                source_flags.setdefault((str(row.ts_code), str(row.report_period)), set()).add(
                    f"tushare.{name}"
                )
            merged = (
                normalized
                if merged is None
                else _merge_financial_report_sources(merged, normalized)
            )
        if merged is None or merged.empty:
            return pd.DataFrame(columns=["ts_code", "report_period"])
        merged["source_flags"] = [
            ",".join(sorted(source_flags.get((str(row.ts_code), str(row.report_period)), set())))
            for row in merged[["ts_code", "report_period"]].itertuples(index=False)
        ]
        merged["pit_safe"] = merged["visible_date"].notna()
        return merged.sort_values(["ts_code", "report_period"]).reset_index(drop=True)

    def _build_financial_current_wide(self, *, snapshot_as_of_date: str | None) -> pd.DataFrame:
        as_of = snapshot_as_of_date or datetime.now(tz=UTC).strftime("%Y%m%d")
        reports = self._build_financial_reports_wide(snapshot_as_of_date=as_of)
        if reports.empty or "visible_date" not in reports.columns:
            return pd.DataFrame(columns=["ts_code", "snapshot_as_of_date"])
        visible = reports[reports["visible_date"].astype(str) <= as_of].copy()
        if visible.empty:
            return pd.DataFrame(columns=["ts_code", "snapshot_as_of_date"])
        visible = visible.sort_values(["ts_code", "visible_date", "report_period"])
        latest = visible.groupby("ts_code", as_index=False).tail(1).copy()
        latest["snapshot_as_of_date"] = as_of
        latest["latest_report_period"] = latest.get("report_period")
        latest["latest_visible_date"] = latest.get("visible_date")
        latest["latest_ann_date"] = latest.get("ann_date")
        latest["source_report_period"] = latest.get("report_period")
        latest["source_visible_date"] = latest.get("visible_date")
        return latest

    def _build_corporate_actions(self, *, snapshot_as_of_date: str | None) -> pd.DataFrame:
        _ = snapshot_as_of_date
        frames: list[pd.DataFrame] = []
        dividend = _read(self.lake, "tushare/dividend")
        if not dividend.empty:
            frame = dividend.copy()
            frame["action_type"] = "dividend"
            frame["source_api"] = "dividend"
            frame["visible_date"] = frame.get("ann_date")
            frame["stock_div"] = frame.get("stk_div")
            frames.append(_corporate_action_payload(frame))
        names = _read(self.lake, "tushare/namechange")
        if not names.empty:
            frame = names.copy()
            frame["action_type"] = "namechange"
            frame["source_api"] = "namechange"
            frame["visible_date"] = frame.get("start_date")
            frame["effective_date"] = frame.get("start_date")
            frames.append(_corporate_action_payload(frame))
        return _concat(frames, ["ts_code", "action_type", "source_api", "source_event_key"])

    def _build_macro_series(self, *, snapshot_as_of_date: str | None) -> pd.DataFrame:
        _ = snapshot_as_of_date
        frames: list[pd.DataFrame] = []
        for api_name, period_column, period_type in (
            ("cn_gdp", "quarter", "quarterly"),
            ("cn_cpi", "month", "monthly"),
            ("cn_ppi", "month", "monthly"),
            ("shibor", "date", "daily"),
        ):
            frame = _read(self.lake, f"tushare/{api_name}")
            if frame.empty or period_column not in frame.columns:
                continue
            value_columns = [column for column in frame.columns if column != period_column]
            melted = frame.melt(
                id_vars=[period_column],
                value_vars=value_columns,
                var_name="source_field",
                value_name="value",
            )
            melted["macro_id"] = api_name + "." + melted["source_field"].astype(str)
            melted["period"] = melted[period_column].astype(str)
            melted["period_type"] = period_type
            melted["visible_date"] = melted["period"]
            melted["source_api"] = api_name
            melted["source"] = "tushare"
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
                        "source",
                    ]
                ]
            )
        return _concat(frames, ["macro_id", "period"])


def _read(lake: DataLake, name: str) -> pd.DataFrame:
    path = lake.dataset_path("raw", name)
    if not path.exists():
        return pd.DataFrame()
    return lake.read_parquet("raw", name)


def _select(frame: pd.DataFrame, mapping: dict[str, str]) -> pd.DataFrame:
    result = pd.DataFrame()
    for source, target in mapping.items():
        if source in frame.columns:
            result[target] = frame[source]
        else:
            result[target] = pd.Series([None] * len(frame), dtype=object)
    return result


def _concat(frames: list[pd.DataFrame], keys: list[str]) -> pd.DataFrame:
    non_empty = [frame for frame in frames if not frame.empty]
    if not non_empty:
        return pd.DataFrame(columns=keys)
    merged = pd.concat(non_empty, ignore_index=True, sort=False)
    available_keys = [key for key in keys if key in merged.columns]
    if available_keys:
        require_unique_keys(
            merged,
            keys=available_keys,
            code="DUPLICATE_TABLE_SOURCE_KEY",
            field="silver_table_source",
        )
        merged = merged.sort_values(available_keys)
    return merged.reset_index(drop=True)


def _corporate_action_payload(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["raw_payload_json"] = [
        json.dumps(record, ensure_ascii=True, sort_keys=True, default=str)
        for record in frame.to_dict(orient="records")
    ]
    frame["event_hash"] = [
        hashlib.sha256(payload.encode()).hexdigest() for payload in frame["raw_payload_json"]
    ]
    frame["source_event_key"] = frame["event_hash"]
    frame["pit_safe"] = True
    frame["source"] = "tushare"
    columns = [
        "ts_code",
        "action_type",
        "source_api",
        "source_event_key",
        "event_hash",
        "ann_date",
        "visible_date",
        "record_date",
        "ex_date",
        "pay_date",
        "effective_date",
        "end_date",
        "cash_div",
        "cash_div_tax",
        "stock_div",
        "change_reason",
        "raw_payload_json",
        "pit_safe",
        "source",
    ]
    for column in columns:
        if column not in frame.columns:
            frame[column] = None
    return frame[columns]


def _normalize_financial_report_frame(frame: pd.DataFrame, *, source_name: str) -> pd.DataFrame:
    data = frame.copy()
    if "_empty" in data.columns:
        data = data.drop(columns=["_empty"])
    if data.empty or "ts_code" not in data.columns or "end_date" not in data.columns:
        return pd.DataFrame()
    data["report_period"] = data["end_date"]
    if "f_ann_date" in data.columns:
        data["visible_date"] = data["f_ann_date"].fillna(data.get("ann_date"))
    elif "ann_date" in data.columns:
        data["visible_date"] = data["ann_date"]
    else:
        data["visible_date"] = None
    if "ann_date" not in data.columns:
        data["ann_date"] = None
    if "report_type" not in data.columns:
        data["report_type"] = None
    data["source_" + source_name] = True
    key_columns = ["ts_code", "report_period"]
    data = (
        data.sort_values([*key_columns, "visible_date", "ann_date"], na_position="first")
        .drop_duplicates(key_columns, keep="last")
        .reset_index(drop=True)
    )
    return data


def _merge_financial_report_sources(left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
    merged = left.merge(
        right,
        on=["ts_code", "report_period"],
        how="outer",
        suffixes=("", "_right"),
    )
    for column in ("ann_date", "visible_date", "report_type"):
        right_column = f"{column}_right"
        if right_column in merged.columns:
            if column in merged.columns:
                merged[column] = merged[column].combine_first(merged[right_column])
            else:
                merged[column] = merged[right_column]
            merged = merged.drop(columns=[right_column])
    duplicate_columns = [
        column
        for column in merged.columns
        if column.endswith("_right") and column.removesuffix("_right") in merged.columns
    ]
    if duplicate_columns:
        merged = merged.drop(columns=duplicate_columns)
    rename_columns = {
        column: column.removesuffix("_right")
        for column in merged.columns
        if column.endswith("_right") and column.removesuffix("_right") not in merged.columns
    }
    if rename_columns:
        merged = merged.rename(columns=rename_columns)
    return merged


def _silver_keys(table: str) -> list[str]:
    return {
        "security_master": ["ts_code"],
        "trade_calendar": ["exchange", "cal_date"],
        "daily_market": ["ts_code", "trade_date"],
        "index_daily": ["ts_code", "trade_date"],
        "financial_reports_wide": ["ts_code", "report_period"],
        "financial_current_wide": ["ts_code"],
        "corporate_actions": ["ts_code", "action_type", "source_api", "source_event_key"],
        "macro_series": ["macro_id", "period"],
    }[table]
