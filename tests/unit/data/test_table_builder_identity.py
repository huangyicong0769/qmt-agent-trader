import pandas as pd
import pytest

from qmt_agent_trader.backtest.errors import BacktestDataIntegrityError
from qmt_agent_trader.data.table_builder import _concat


def test_table_builder_concat_rejects_duplicate_business_key() -> None:
    frames = [
        pd.DataFrame([{"ts_code": "000001.SZ", "name": "first"}]),
        pd.DataFrame([{"ts_code": "000001.SZ", "name": "second"}]),
    ]

    with pytest.raises(BacktestDataIntegrityError) as exc_info:
        _concat(frames, ["ts_code"])

    assert exc_info.value.code == "DUPLICATE_TABLE_SOURCE_KEY"
