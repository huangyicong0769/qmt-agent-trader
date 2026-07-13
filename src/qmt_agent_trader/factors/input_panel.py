"""Build target-frequency factor input panels with PIT-safe joins."""

from __future__ import annotations

from datetime import date, datetime
from typing import Any

import pandas as pd

from qmt_agent_trader.data.bars import load_daily_bars
from qmt_agent_trader.data.field_sources import FieldSourceIndex, FieldSourceSpec, FillPolicy
from qmt_agent_trader.data.frequency import Frequency
from qmt_agent_trader.data.macro import get_macro_dataset, macro_visible_date
from qmt_agent_trader.data.providers.tushare.registry import default_tushare_registry
from qmt_agent_trader.data.storage import DataLake


def build_target_frequency_panel(
    lake: DataLake,
    *,
    target_frequency: Frequency,
    target_start: str | date,
    target_end: str | date,
    required_fields: list[str],
    symbols: list[str] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    start = _parse_date(target_start)
    end = _parse_date(target_end)
    metadata = _base_metadata(
        target_frequency=target_frequency,
        target_start=start,
        target_end=end,
        required_fields=required_fields,
    )
    if target_frequency is not Frequency.DAILY:
        metadata["status"] = "INVALID_REQUEST"
        metadata["warnings"].append("only daily target-frequency panels are implemented")
        return pd.DataFrame(), metadata

    panel = load_daily_bars(
        lake,
        start=_date_key(start),
        end=_date_key(end),
        symbols=symbols,
    )
    if panel.empty:
        metadata["status"] = "NO_DATA"
        metadata["missing_fields"]["__skeleton__"] = {
            "reason": "daily_bars_missing",
            "suggested_next_step": "fetch tushare daily or fund_daily for the target range",
        }
        return panel, metadata

    panel = panel.copy()
    panel["trade_date"] = _coerce_date(panel["trade_date"])
    panel = panel.sort_values(["symbol", "trade_date"]).reset_index(drop=True)
    source_index = FieldSourceIndex.from_tushare_registry(default_tushare_registry())
    for field in dict.fromkeys(required_fields):
        if field in panel.columns:
            continue
        source = source_index.best_source_for_field(field, target_frequency=target_frequency)
        candidates = source_index.sources_for_field(field)
        if source is None:
            _record_unresolved_source(metadata, field, candidates)
            continue
        metadata["field_sources"][field] = source.as_metadata()
        if source.fill_policy is FillPolicy.EXACT:
            panel = _join_exact_field(
                lake,
                panel,
                source=source,
                field=field,
                start=start,
                end=end,
                symbols=symbols,
                metadata=metadata,
            )
        elif source.fill_policy is FillPolicy.ASOF_SNAPSHOT:
            panel = _join_asof_snapshot_field(
                lake,
                panel,
                source=source,
                field=field,
                end=end,
                symbols=symbols,
                metadata=metadata,
            )
        elif source.fill_policy is FillPolicy.EVENT_TO_STATE:
            metadata["unresolved_fields"].append(
                {
                    "field": field,
                    "api_name": source.api_name,
                    "status": "NOT_IMPLEMENTED_FOR_EVENT_STATE",
                    "reason": "event_to_state_not_implemented",
                }
            )
        else:
            metadata["unresolved_fields"].append(
                {
                    "field": field,
                    "api_name": source.api_name,
                    "status": "UNRESOLVED_FIELD",
                    "reason": "event_field_requires_explicit_transform",
                    "suggested_next_step": (
                        "implement event_to_state transform or event-window factor"
                    ),
                }
            )

    metadata["coverage_by_field"] = _coverage_by_field(panel, required_fields, metadata)
    metadata.update(_daily_panel_coverage(panel))
    if metadata["unresolved_fields"]:
        metadata["status"] = "PARTIAL_COVERAGE"
    elif metadata["missing_fields"]:
        metadata["status"] = "PARTIAL_COVERAGE"
    else:
        metadata["status"] = "OK"
    metadata["input_panel_status"] = metadata["status"]
    metadata["evidence_status"] = "STRONG" if metadata["status"] == "OK" else "BLOCKED"
    return panel, metadata


def _join_exact_field(
    lake: DataLake,
    panel: pd.DataFrame,
    *,
    source: FieldSourceSpec,
    field: str,
    start: date,
    end: date,
    symbols: list[str] | None,
    metadata: dict[str, Any],
) -> pd.DataFrame:
    if source.entity_column is None:
        _record_unresolved(metadata, field, source, "daily_marketwide_exact_not_supported")
        return panel
    if source.visible_time_column is None:
        _record_unresolved(metadata, field, source, "visible_time_column_missing")
        return panel
    if not lake.dataset_path("raw", source.raw_dataset_name).exists():
        return _add_missing_field(
            panel,
            metadata,
            field,
            source,
            reason="raw_dataset_missing",
        )

    columns = _source_read_columns(source, field)
    raw = lake.read_parquet_filtered(
        "raw",
        source.raw_dataset_name,
        columns=columns,
        start=start,
        end=end,
        date_column=source.visible_time_column,
        symbols=symbols,
        symbol_column=source.entity_column,
    )
    missing_columns = [
        column
        for column in (source.entity_column, source.visible_time_column, field)
        if column is not None and column not in raw.columns
    ]
    if missing_columns:
        return _add_missing_field(
            panel,
            metadata,
            field,
            source,
            reason="raw_field_missing",
            details={"missing_columns": missing_columns},
        )
    if raw.empty:
        return _add_missing_field(panel, metadata, field, source, reason="no_source_rows")

    data = raw.rename(columns={source.entity_column: "symbol"}).copy()
    data["trade_date"] = _coerce_date(data[source.visible_time_column])
    data = (
        data[["symbol", "trade_date", field]]
        .dropna(subset=["symbol", "trade_date"])
        .drop_duplicates(["symbol", "trade_date"], keep="last")
    )
    joined = panel.merge(data, on=["symbol", "trade_date"], how="left")
    return joined


def _join_asof_snapshot_field(
    lake: DataLake,
    panel: pd.DataFrame,
    *,
    source: FieldSourceSpec,
    field: str,
    end: date,
    symbols: list[str] | None,
    metadata: dict[str, Any],
) -> pd.DataFrame:
    if not lake.dataset_path("raw", source.raw_dataset_name).exists():
        return _add_missing_field(
            panel,
            metadata,
            field,
            source,
            reason="raw_dataset_missing",
        )
    raw = lake.read_parquet("raw", source.raw_dataset_name)
    if field not in raw.columns:
        return _add_missing_field(panel, metadata, field, source, reason="raw_field_missing")
    data = raw.copy()
    if source.entity_column is not None:
        if source.entity_column not in data.columns:
            return _add_missing_field(
                panel,
                metadata,
                field,
                source,
                reason="raw_field_missing",
                details={"missing_columns": [source.entity_column]},
            )
        data = data.rename(columns={source.entity_column: "symbol"})
        if symbols:
            data = data[data["symbol"].astype(str).isin(symbols)]

    visible = _visible_dates(data, source)
    if visible is None:
        _record_unresolved(metadata, field, source, "visible_time_column_missing")
        return panel
    data = data.copy()
    data["visible_date"] = visible
    data = data[data["visible_date"].notna()]
    data = data[data["visible_date"] <= end]
    if data.empty:
        return _add_missing_field(panel, metadata, field, source, reason="no_source_rows")
    if source.entity_column is None:
        return _join_marketwide_asof(panel, data, field)
    return _join_symbol_asof(panel, data, field)


def _join_symbol_asof(panel: pd.DataFrame, data: pd.DataFrame, field: str) -> pd.DataFrame:
    left = panel[["symbol", "trade_date"]].copy()
    left["_panel_row_id"] = range(len(left))
    left["_trade_ts"] = pd.to_datetime(left["trade_date"], errors="coerce")
    right = (
        data[["symbol", "visible_date", field]]
        .dropna(subset=["symbol", "visible_date"])
        .sort_values(["symbol", "visible_date"])
        .drop_duplicates(["symbol", "visible_date"], keep="last")
    )
    right = right.copy()
    right["_visible_ts"] = pd.to_datetime(right["visible_date"], errors="coerce")
    pieces: list[pd.DataFrame] = []
    for symbol, left_group in left.groupby("symbol", sort=False):
        right_group = right[right["symbol"] == symbol]
        if right_group.empty:
            piece = left_group[["_panel_row_id"]].copy()
            piece[field] = pd.NA
            pieces.append(piece)
            continue
        merged = pd.merge_asof(
            left_group.sort_values("_trade_ts"),
            right_group[["_visible_ts", field]].sort_values("_visible_ts"),
            left_on="_trade_ts",
            right_on="_visible_ts",
            direction="backward",
        )
        pieces.append(merged[["_panel_row_id", field]])
    values = pd.concat(pieces, ignore_index=True).set_index("_panel_row_id")[field]
    result = panel.copy()
    result[field] = values.reindex(range(len(result))).reset_index(drop=True)
    return result


def _join_marketwide_asof(panel: pd.DataFrame, data: pd.DataFrame, field: str) -> pd.DataFrame:
    left = panel[["trade_date"]].drop_duplicates().sort_values("trade_date")
    left = left.copy()
    left["_trade_ts"] = pd.to_datetime(left["trade_date"], errors="coerce")
    right = (
        data[["visible_date", field]]
        .dropna(subset=["visible_date"])
        .sort_values("visible_date")
        .drop_duplicates(["visible_date"], keep="last")
    )
    right = right.copy()
    right["_visible_ts"] = pd.to_datetime(right["visible_date"], errors="coerce")
    merged = pd.merge_asof(
        left.sort_values("_trade_ts"),
        right[["_visible_ts", field]].sort_values("_visible_ts"),
        left_on="_trade_ts",
        right_on="_visible_ts",
        direction="backward",
    )
    return panel.merge(merged[["trade_date", field]], on="trade_date", how="left")


def _visible_dates(data: pd.DataFrame, source: FieldSourceSpec) -> pd.Series | None:
    macro = get_macro_dataset(source.api_name)
    if macro is not None:
        if macro.date_column not in data.columns:
            return None
        return pd.Series(macro_visible_date(data[macro.date_column], macro), index=data.index)
    if source.visible_time_column and source.visible_time_column in data.columns:
        visible = _coerce_date(data[source.visible_time_column])
        if source.fallback_visible_time_column and source.fallback_visible_time_column in data:
            fallback = _coerce_date(data[source.fallback_visible_time_column])
            visible = visible.where(pd.Series(visible).notna(), fallback)
        return pd.Series(visible, index=data.index)
    if source.fallback_visible_time_column and source.fallback_visible_time_column in data.columns:
        return pd.Series(_coerce_date(data[source.fallback_visible_time_column]), index=data.index)
    return None


def _record_unresolved_source(
    metadata: dict[str, Any],
    field: str,
    candidates: list[FieldSourceSpec],
) -> None:
    if not candidates:
        metadata["unresolved_fields"].append(
            {
                "field": field,
                "status": "UNRESOLVED_FIELD",
                "reason": "field_source_not_found",
                "suggested_next_step": "run capability discovery for the field",
            }
        )
        return
    metadata["unresolved_fields"].append(
        {
            "field": field,
            "status": "AMBIGUOUS_FIELD_SOURCE",
            "reason": "multiple_candidate_sources",
            "candidates": [
                {
                    "api_name": source.api_name,
                    "raw_dataset_name": source.raw_dataset_name,
                    "frequency": source.frequency.value,
                    "fill_policy": source.fill_policy.value,
                    "pit_safe": source.pit_safe,
                }
                for source in candidates
            ],
        }
    )


def _record_unresolved(
    metadata: dict[str, Any],
    field: str,
    source: FieldSourceSpec,
    reason: str,
) -> None:
    metadata["unresolved_fields"].append(
        {
            "field": field,
            "api_name": source.api_name,
            "status": "UNRESOLVED_FIELD",
            "reason": reason,
        }
    )


def _add_missing_field(
    panel: pd.DataFrame,
    metadata: dict[str, Any],
    field: str,
    source: FieldSourceSpec,
    *,
    reason: str,
    details: dict[str, Any] | None = None,
) -> pd.DataFrame:
    result = panel.copy()
    if field not in result.columns:
        result[field] = pd.NA
    payload: dict[str, Any] = {
        "reason": reason,
        "api_name": source.api_name,
        "raw_dataset_name": source.raw_dataset_name,
    }
    if details:
        payload.update(details)
    metadata["missing_fields"][field] = payload
    return result


def _coverage_by_field(
    panel: pd.DataFrame,
    fields: list[str],
    metadata: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    coverage: dict[str, dict[str, Any]] = {}
    row_count = len(panel)
    field_sources = metadata.get("field_sources")
    field_sources = field_sources if isinstance(field_sources, dict) else {}
    for field in dict.fromkeys(fields):
        if field in panel.columns:
            series = panel[field]
        else:
            series = pd.Series([pd.NA] * row_count)
        non_null = int(series.notna().sum())
        dates = panel.loc[series.notna(), "trade_date"] if field in panel.columns else []
        item: dict[str, Any] = {
            "non_null": non_null,
            "non_null_rows": non_null,
            "total_rows": row_count,
            "coverage": non_null / row_count if row_count else 0.0,
            "first_available_date": _format_date(min(dates)) if len(dates) else None,
            "last_available_date": _format_date(max(dates)) if len(dates) else None,
        }
        source = field_sources.get(field)
        if isinstance(source, dict):
            raw_dataset = str(source.get("raw_dataset_name") or "")
            item["source"] = f"raw/{raw_dataset}" if raw_dataset else None
            item["join_policy"] = source.get("fill_policy")
            item["pit_safe"] = bool(source.get("pit_safe"))
        coverage[field] = item
    return coverage


def _daily_panel_coverage(panel: pd.DataFrame) -> dict[str, object]:
    if panel.empty or not {"trade_date", "symbol"}.issubset(panel.columns):
        return {
            "daily_row_counts": {},
            "daily_symbol_counts": {},
            "daily_cross_sectional_coverage": {},
            "daily_reference_symbol_counts": {},
        }
    row_counts = panel.groupby("trade_date").size().sort_index().astype(int)
    symbol_counts = panel.groupby("trade_date")["symbol"].nunique().sort_index().astype(int)
    prior_reference = symbol_counts.shift(1).rolling(20, min_periods=5).median()
    ratios = symbol_counts / prior_reference.replace(0, pd.NA)
    return {
        "daily_row_counts": {_format_date(index): int(value) for index, value in row_counts.items()},
        "daily_symbol_counts": {
            _format_date(index): int(value) for index, value in symbol_counts.items()
        },
        "daily_cross_sectional_coverage": {
            _format_date(index): float(value) for index, value in ratios.dropna().items()
        },
        "daily_reference_symbol_counts": {
            _format_date(index): float(value)
            for index, value in prior_reference.dropna().items()
        },
    }


def _base_metadata(
    *,
    target_frequency: Frequency,
    target_start: date,
    target_end: date,
    required_fields: list[str],
) -> dict[str, Any]:
    return {
        "status": "UNKNOWN",
        "target_frequency": target_frequency.value,
        "target_start": _date_key(target_start),
        "target_end": _date_key(target_end),
        "required_fields": list(required_fields),
        "field_sources": {},
        "coverage_by_field": {},
        "daily_row_counts": {},
        "daily_symbol_counts": {},
        "daily_cross_sectional_coverage": {},
        "daily_reference_symbol_counts": {},
        "abrupt_low_coverage_dates": [],
        "missing_fields": {},
        "unresolved_fields": [],
        "warnings": [],
    }


def _source_read_columns(source: FieldSourceSpec, field: str) -> list[str]:
    columns = [
        source.entity_column,
        source.source_time_column,
        source.visible_time_column,
        source.fallback_visible_time_column,
        field,
    ]
    return list(dict.fromkeys(column for column in columns if column is not None))


def _coerce_date(values: pd.Series) -> pd.Series:
    if values.empty:
        return pd.Series([], dtype="object")
    text = values.astype(str).str.strip()
    text = text.replace({"": None, "NaT": None, "None": None, "nan": None, "<NA>": None})
    parsed = pd.to_datetime(text, format="%Y%m%d", errors="coerce")
    missing = parsed.isna()
    if missing.any():
        parsed.loc[missing] = pd.to_datetime(text[missing], errors="coerce")
    return parsed.dt.date


def _parse_date(value: str | date) -> date:
    if isinstance(value, date):
        return value
    for fmt in ("%Y%m%d", "%Y-%m-%d"):
        try:
            return datetime.strptime(value, fmt).date()
        except ValueError:
            continue
    return datetime.fromisoformat(value).date()


def _date_key(value: date) -> str:
    return f"{value:%Y%m%d}"


def _format_date(value: Any) -> str:
    if isinstance(value, date):
        return f"{value:%Y-%m-%d}"
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        return str(value)
    return f"{parsed.date():%Y-%m-%d}"
