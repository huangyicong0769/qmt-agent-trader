import pandas as pd
import pytest

from qmt_agent_trader.backtest.errors import BacktestDataIntegrityError
from qmt_agent_trader.data.transforms.universe import filter_tradeable_universe


def test_universe_filter_rejects_missing_state() -> None:
    with pytest.raises(BacktestDataIntegrityError) as exc_info:
        filter_tradeable_universe(pd.DataFrame([{"symbol": "A"}]))

    assert exc_info.value.code == "MISSING_EXECUTION_STATE_COLUMNS"


def test_universe_filter_rejects_unknown_state() -> None:
    frame = pd.DataFrame([{"symbol": "A", "st": None, "suspended": False}])

    with pytest.raises(BacktestDataIntegrityError) as exc_info:
        filter_tradeable_universe(frame)

    assert exc_info.value.code == "UNKNOWN_EXECUTION_STATE"
