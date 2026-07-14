"""Fail-closed validation for tabular data identities."""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

from qmt_agent_trader.backtest.errors import BacktestDataIntegrityError


def require_unique_keys(
    frame: pd.DataFrame,
    *,
    keys: Sequence[str],
    code: str,
    field: str,
) -> None:
    missing = [key for key in keys if key not in frame.columns]
    if missing:
        raise BacktestDataIntegrityError(
            code="INVALID_IDENTITY_FRAME",
            message="identity validation requires missing columns",
            field=field,
            details={"missing_columns": missing},
        )
    duplicate_mask = frame.duplicated(list(keys), keep=False)
    if not duplicate_mask.any():
        return
    duplicate_keys = (
        frame.loc[duplicate_mask, list(keys)]
        .drop_duplicates()
        .sort_values(list(keys), kind="stable")
    )
    symbols: tuple[str, ...] = ()
    for symbol_column in ("symbol", "ts_code", "con_code"):
        if symbol_column in duplicate_keys.columns:
            symbols = tuple(
                sorted(duplicate_keys[symbol_column].astype(str).unique().tolist())
            )
            break
    sample = [
        {key: _jsonable_identity_value(getattr(row, key)) for key in keys}
        for row in duplicate_keys.head(20).itertuples(index=False)
    ]
    raise BacktestDataIntegrityError(
        code=code,
        message="identity columns must be unique",
        symbols=symbols,
        field=field,
        details={
            "identity_columns": list(keys),
            "duplicate_key_count": len(duplicate_keys),
            "sample": sample,
        },
    )


def require_unique_symbol_dates(
    frame: pd.DataFrame,
    *,
    symbol_column: str,
    date_column: str,
    code: str,
    field: str,
) -> None:
    require_unique_keys(
        frame,
        keys=(symbol_column, date_column),
        code=code,
        field=field,
    )


def _jsonable_identity_value(value: object) -> object:
    if isinstance(value, pd.Timestamp):
        return value.date().isoformat()
    if hasattr(value, "isoformat"):
        try:
            return value.isoformat()
        except TypeError:
            pass
    return str(value)
