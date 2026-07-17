"""Point-in-time metadata used by universe resolution."""

from __future__ import annotations

from datetime import date

import pandas as pd

from qmt_agent_trader.backtest.errors import BacktestUniverseIntegrityError
from qmt_agent_trader.data.integrity import require_unique_keys
from qmt_agent_trader.universe.validators import normalize_symbol


def _normalized_member_values(
    values: pd.Series,
    *,
    field: str,
) -> list[str]:
    normalized: list[str] = []
    invalid_count = 0
    for raw in values.tolist():
        text = "" if raw is None else str(raw).strip()
        symbol = normalize_symbol(text) if text else None
        if symbol is None:
            invalid_count += 1
            continue
        if symbol not in normalized:
            normalized.append(symbol)
    if invalid_count:
        raise BacktestUniverseIntegrityError(
            code="INDEX_MEMBERSHIP_SOURCE_INVALID",
            message="index membership contains invalid member identifiers",
            field=field,
            details={
                "invalid_row_count": invalid_count,
            },
        )
    return sorted(normalized)


def security_master_asof(
    stock_basic: pd.DataFrame,
    as_of_date: date,
) -> pd.DataFrame:
    if stock_basic.empty:
        return pd.DataFrame(
            columns=[
                "symbol",
                "display_name",
                "list_date",
                "delist_date",
                "listed_as_of",
            ]
        )
    required = {"ts_code", "list_date"}
    missing = sorted(required.difference(stock_basic.columns))
    if missing:
        raise BacktestUniverseIntegrityError(
            code="UNIVERSE_SECURITY_MASTER_INVALID",
            message="stock_basic lacks listing-window fields",
            field="raw/tushare/stock_basic",
            details={"missing_columns": missing},
        )
    data = stock_basic.copy()
    require_unique_keys(
        data,
        keys=("ts_code",),
        code="DUPLICATE_UNIVERSE_SOURCE_KEY",
        field="raw/tushare/stock_basic",
    )
    data["symbol"] = data["ts_code"].astype(str)
    data["list_date"] = _required_date(
        data["list_date"],
        field="raw/tushare/stock_basic.list_date",
        error_code="UNIVERSE_SECURITY_MASTER_INVALID",
    )
    if "delist_date" in data.columns:
        data["delist_date"] = _optional_date(
            data["delist_date"],
            field="raw/tushare/stock_basic.delist_date",
            error_code="UNIVERSE_SECURITY_MASTER_INVALID",
        )
    else:
        data["delist_date"] = pd.NaT
    data["display_name"] = (
        data["name"].astype("string")
        if "name" in data.columns
        else pd.Series(pd.NA, index=data.index, dtype="string")
    )
    listed_before_boundary = data["list_date"].map(
        lambda value: pd.notna(value) and value <= as_of_date
    )
    active_before_delist = data["delist_date"].map(
        lambda value: pd.isna(value) or value > as_of_date
    )
    data["listed_as_of"] = listed_before_boundary & active_before_delist
    return data[
        [
            "symbol",
            "display_name",
            "list_date",
            "delist_date",
            "listed_as_of",
        ]
    ]


def require_historical_classification_support(
    *,
    selection_mode: str,
    as_of_date: date,
    classification_frame: pd.DataFrame | None,
) -> None:
    if selection_mode not in {"industry", "theme"}:
        return
    required = {"symbol", "effective_from", "effective_to"}
    available = (
        set(classification_frame.columns)
        if classification_frame is not None
        else set()
    )
    if not required.issubset(available):
        raise BacktestUniverseIntegrityError(
            code="UNIVERSE_PIT_CLASSIFICATION_NOT_READY",
            message=(
                "historical industry/theme selection requires dated "
                "classification evidence"
            ),
            trade_date=as_of_date.isoformat(),
            field="classification_history",
            details={"selection_mode": selection_mode},
        )


def index_weight_members_by_code_asof(
    frame: pd.DataFrame,
    index_codes: list[str],
    as_of: date,
) -> dict[str, list[str]]:
    required = {"index_code", "con_code", "trade_date"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise BacktestUniverseIntegrityError(
            code="INDEX_MEMBERSHIP_SOURCE_INVALID",
            message="index_weight lacks required columns",
            field="raw/tushare/index_weight",
            details={"missing_columns": missing},
        )
    data = frame.copy()
    data["index_code"] = data["index_code"].astype(str)
    data["trade_date"] = _required_date(
        data["trade_date"],
        field="raw/tushare/index_weight.trade_date",
        error_code="INDEX_MEMBERSHIP_SOURCE_INVALID",
    )
    requested = set(index_codes)
    data = data[
        data["index_code"].isin(requested)
        & data["trade_date"].map(lambda value: value <= as_of)
    ]
    result: dict[str, list[str]] = {}
    for index_code, group in data.groupby("index_code", sort=True):
        snapshot_date = group["trade_date"].max()
        snapshot = group[group["trade_date"].eq(snapshot_date)]
        require_unique_keys(
            snapshot,
            keys=("index_code", "con_code", "trade_date"),
            code="DUPLICATE_UNIVERSE_SOURCE_KEY",
            field="raw/tushare/index_weight",
        )
        members = _normalized_member_values(
            snapshot["con_code"],
            field="raw/tushare/index_weight.con_code",
        )
        if members:
            result[str(index_code)] = members
    return result


def index_weight_members_asof(
    frame: pd.DataFrame,
    index_codes: list[str],
    as_of: date,
) -> list[str]:
    grouped = index_weight_members_by_code_asof(frame, index_codes, as_of)
    return sorted(
        {symbol for members in grouped.values() for symbol in members}
    )


def index_interval_members_by_code_asof(
    frame: pd.DataFrame,
    index_codes: list[str],
    as_of: date,
) -> dict[str, list[str]]:
    required = {"index_code", "con_code", "in_date", "out_date"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise BacktestUniverseIntegrityError(
            code="INDEX_MEMBERSHIP_SOURCE_INVALID",
            message="index_member lacks effective interval columns",
            field="raw/tushare/index_member",
            details={"missing_columns": missing},
        )
    data = frame.copy()
    data["index_code"] = data["index_code"].astype(str)
    data["in_date"] = _required_date(
        data["in_date"],
        field="raw/tushare/index_member.in_date",
        error_code="INDEX_MEMBERSHIP_SOURCE_INVALID",
    )
    data["out_date"] = _optional_date(
        data["out_date"],
        field="raw/tushare/index_member.out_date",
        error_code="INDEX_MEMBERSHIP_SOURCE_INVALID",
    )
    requested = set(index_codes)
    active = data[
        data["index_code"].isin(requested)
        & data["in_date"].map(lambda value: value <= as_of)
        & data["out_date"].map(lambda value: pd.isna(value) or value > as_of)
    ]
    require_unique_keys(
        active,
        keys=("index_code", "con_code"),
        code="DUPLICATE_UNIVERSE_SOURCE_KEY",
        field="raw/tushare/index_member",
    )
    result: dict[str, list[str]] = {}
    for code in sorted(requested):
        code_active = active[active["index_code"].eq(code)]
        if code_active.empty:
            continue
        members = _normalized_member_values(
            code_active["con_code"],
            field="raw/tushare/index_member.con_code",
        )
        if members:
            result[code] = members
    return result


def index_interval_members_asof(
    frame: pd.DataFrame,
    index_codes: list[str],
    as_of: date,
) -> list[str]:
    grouped = index_interval_members_by_code_asof(frame, index_codes, as_of)
    return sorted(
        {symbol for members in grouped.values() for symbol in members}
    )


_MISSING_DATE_TOKENS = {"", "nan", "nat", "none", "<na>"}


def _date_text(values: pd.Series) -> tuple[pd.Series, pd.Series]:
    text = values.astype("string").str.strip()
    missing = values.isna() | text.str.lower().isin(_MISSING_DATE_TOKENS)
    return text, missing


def _required_date(
    values: pd.Series,
    *,
    field: str,
    error_code: str,
) -> pd.Series:
    text, missing = _date_text(values)
    parsed = pd.to_datetime(
        text.where(~missing),
        format="mixed",
        errors="coerce",
    )
    invalid = missing | parsed.isna()
    if invalid.any():
        raise BacktestUniverseIntegrityError(
            code=error_code,
            message="point-in-time source contains an invalid required date",
            field=field,
            details={"invalid_row_count": int(invalid.sum())},
        )
    return parsed.dt.date


def _optional_date(
    values: pd.Series,
    *,
    field: str,
    error_code: str,
) -> pd.Series:
    text, missing = _date_text(values)
    parsed = pd.to_datetime(
        text.where(~missing),
        format="mixed",
        errors="coerce",
    )
    invalid = ~missing & parsed.isna()
    if invalid.any():
        raise BacktestUniverseIntegrityError(
            code=error_code,
            message="point-in-time source contains an invalid optional date",
            field=field,
            details={"invalid_row_count": int(invalid.sum())},
        )
    return parsed.dt.date
