from datetime import date

import pandas as pd
import pytest

from qmt_agent_trader.backtest.errors import BacktestDataIntegrityError
from qmt_agent_trader.factors.input_panel import (
    _join_marketwide_asof,
    _join_symbol_asof,
)


def test_symbol_asof_duplicate_visible_key_fails_closed() -> None:
    panel = pd.DataFrame([{"symbol": "A", "trade_date": date(2024, 1, 5)}])
    source = pd.DataFrame(
        [
            {"symbol": "A", "visible_date": date(2024, 1, 4), "roe": 0.10},
            {"symbol": "A", "visible_date": date(2024, 1, 4), "roe": 0.12},
        ]
    )

    with pytest.raises(BacktestDataIntegrityError) as exc_info:
        _join_symbol_asof(panel, source, "roe")

    assert exc_info.value.code == "DUPLICATE_ASOF_VISIBLE_KEY"


def test_marketwide_asof_duplicate_visible_key_fails_closed() -> None:
    panel = pd.DataFrame([{"symbol": "A", "trade_date": date(2024, 1, 5)}])
    source = pd.DataFrame(
        [
            {"visible_date": date(2024, 1, 4), "pmi": 49.0},
            {"visible_date": date(2024, 1, 4), "pmi": 50.0},
        ]
    )

    with pytest.raises(BacktestDataIntegrityError) as exc_info:
        _join_marketwide_asof(panel, source, "pmi")

    assert exc_info.value.code == "DUPLICATE_ASOF_VISIBLE_KEY"
