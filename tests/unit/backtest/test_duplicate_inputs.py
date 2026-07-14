from datetime import date

import pandas as pd
import pytest

from qmt_agent_trader.backtest.errors import BacktestDataIntegrityError
from qmt_agent_trader.backtest.research_runner import (
    FactorRankResearchConfig,
    FactorRankResearchRunner,
)


def _bar(*, close: float = 10.0) -> dict[str, object]:
    return {
        "symbol": "000001.SZ",
        "trade_date": date(2024, 1, 2),
        "open": 10.0,
        "close": close,
    }


def _config() -> FactorRankResearchConfig:
    return FactorRankResearchConfig(
        factor_name="fixture",
        expected_trade_dates=(date(2024, 1, 2),),
    )


@pytest.mark.parametrize("second_close", [10.0, 11.0])
def test_duplicate_bar_symbol_date_is_rejected(second_close) -> None:
    bars = pd.DataFrame([_bar(), _bar(close=second_close)])

    with pytest.raises(BacktestDataIntegrityError) as exc_info:
        FactorRankResearchRunner(bars, _config())

    assert exc_info.value.code == "DUPLICATE_SYMBOL_DATE_BAR"
    assert exc_info.value.details["duplicate_key_count"] == 1


def test_duplicate_factor_symbol_date_is_rejected(monkeypatch) -> None:
    factor_frame = pd.DataFrame(
        [
            {
                "symbol": "000001.SZ",
                "trade_date": date(2024, 1, 2),
                "factor_value": 1.0,
            },
            {
                "symbol": "000001.SZ",
                "trade_date": date(2024, 1, 2),
                "factor_value": 2.0,
            },
        ]
    )
    monkeypatch.setattr(
        "qmt_agent_trader.backtest.research_runner.compute_factor_frame",
        lambda *_args, **_kwargs: factor_frame,
    )

    with pytest.raises(BacktestDataIntegrityError) as exc_info:
        FactorRankResearchRunner(pd.DataFrame([_bar()]), _config())

    assert exc_info.value.code == "DUPLICATE_FACTOR_SYMBOL_DATE"
    assert exc_info.value.symbols == ("000001.SZ",)
