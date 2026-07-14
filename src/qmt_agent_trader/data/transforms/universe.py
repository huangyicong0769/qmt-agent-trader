"""Universe filters."""

from __future__ import annotations

import pandas as pd

from qmt_agent_trader.backtest.errors import BacktestDataIntegrityError


def filter_tradeable_universe(frame: pd.DataFrame) -> pd.DataFrame:
    required = ("st", "suspended")
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise BacktestDataIntegrityError(
            code="MISSING_EXECUTION_STATE_COLUMNS",
            message="universe filter requires validated execution state",
            field="trade_state",
            details={"missing_columns": missing},
        )
    for column in required:
        if frame[column].isna().any():
            raise BacktestDataIntegrityError(
                code="UNKNOWN_EXECUTION_STATE",
                message="universe filter received unknown execution state",
                field=column,
            )
    mask = ~frame["st"].astype(bool) & ~frame["suspended"].astype(bool)
    return frame[mask].copy()
