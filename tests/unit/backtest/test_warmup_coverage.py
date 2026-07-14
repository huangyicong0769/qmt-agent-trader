from datetime import date

import pandas as pd
import pytest

from qmt_agent_trader.backtest.errors import BacktestDataIntegrityError
from qmt_agent_trader.strategy.execution_adapter import _validate_warmup_panel


def test_missing_whole_warmup_session_fails_closed() -> None:
    panel = pd.DataFrame(
        [
            {"symbol": "000001.SZ", "trade_date": date(2024, 1, 2)},
            {"symbol": "000001.SZ", "trade_date": date(2024, 1, 4)},
        ]
    )

    with pytest.raises(BacktestDataIntegrityError) as exc_info:
        _validate_warmup_panel(
            panel,
            warmup_dates=(date(2024, 1, 2), date(2024, 1, 3)),
            expected_trade_dates=(date(2024, 1, 4),),
            required_symbols=("000001.SZ",),
            lookback_sessions=2,
        )

    assert exc_info.value.code == "MISSING_FACTOR_WARMUP_SESSION"


def test_insufficient_symbol_history_is_explicit() -> None:
    panel = pd.DataFrame(
        [
            {"symbol": "A", "trade_date": date(2024, 1, 2)},
            {"symbol": "A", "trade_date": date(2024, 1, 3)},
            {"symbol": "B", "trade_date": date(2024, 1, 3)},
            {"symbol": "A", "trade_date": date(2024, 1, 4)},
            {"symbol": "B", "trade_date": date(2024, 1, 4)},
        ]
    )

    quality = _validate_warmup_panel(
        panel,
        warmup_dates=(date(2024, 1, 2), date(2024, 1, 3)),
        expected_trade_dates=(date(2024, 1, 4),),
        required_symbols=("A", "B"),
        lookback_sessions=2,
    )

    assert quality["insufficient_history_by_symbol"] == {
        "B": {"observed_sessions": 1, "required_sessions": 2}
    }
