from datetime import date

import pandas as pd
import pytest

from qmt_agent_trader.backtest.errors import BacktestDataIntegrityError
from qmt_agent_trader.backtest.research_runner import _prepare_bars


def canonical_row() -> dict[str, object]:
    return {
        "symbol": "000001.SZ",
        "trade_date": date(2024, 1, 2),
        "open": 10.0,
        "high": 10.2,
        "low": 9.8,
        "close": 10.1,
        "volume": 100.0,
        "amount": 1_000.0,
        "turnover": 0.01,
        "suspended": False,
        "st": False,
        "limit_up_at_open": False,
        "limit_down_at_open": False,
    }


def test_missing_execution_state_column_fails_closed() -> None:
    row = canonical_row()
    row.pop("suspended")

    with pytest.raises(BacktestDataIntegrityError) as exc_info:
        _prepare_bars(pd.DataFrame([row]))

    assert exc_info.value.code == "MISSING_EXECUTION_STATE_COLUMNS"
    assert exc_info.value.details["missing_columns"] == ["suspended"]


def test_null_execution_state_fails_closed() -> None:
    row = canonical_row()
    row["limit_up_at_open"] = None

    with pytest.raises(BacktestDataIntegrityError) as exc_info:
        _prepare_bars(pd.DataFrame([row]))

    assert exc_info.value.code == "UNKNOWN_EXECUTION_STATE"
    assert exc_info.value.field == "limit_up_at_open"


def test_missing_numeric_input_is_not_synthesized() -> None:
    row = canonical_row()
    row.pop("turnover")

    with pytest.raises(BacktestDataIntegrityError) as exc_info:
        _prepare_bars(pd.DataFrame([row]))

    assert exc_info.value.code == "MISSING_CANONICAL_BAR_COLUMNS"
    assert exc_info.value.details["missing_columns"] == ["turnover"]
