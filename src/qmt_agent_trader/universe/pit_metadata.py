"""Point-in-time metadata used by universe resolution."""

from __future__ import annotations

from datetime import date

import pandas as pd

from qmt_agent_trader.backtest.errors import BacktestUniverseIntegrityError
from qmt_agent_trader.data.integrity import require_unique_keys


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
    )
    if "delist_date" in data.columns:
        data["delist_date"] = _optional_date(data["delist_date"])
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


def _required_date(values: pd.Series, *, field: str) -> pd.Series:
    parsed = pd.to_datetime(
        values.astype("string"), format="mixed", errors="coerce"
    ).dt.date
    if parsed.isna().any():
        raise BacktestUniverseIntegrityError(
            code="UNIVERSE_SECURITY_MASTER_INVALID",
            message="security master contains invalid listing dates",
            field=field,
            details={"invalid_row_count": int(parsed.isna().sum())},
        )
    return parsed


def _optional_date(values: pd.Series) -> pd.Series:
    return pd.to_datetime(
        values.astype("string"), format="mixed", errors="coerce"
    ).dt.date
