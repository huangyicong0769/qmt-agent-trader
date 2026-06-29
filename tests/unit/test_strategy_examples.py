from datetime import date, timedelta

import pandas as pd

from qmt_agent_trader.strategy.examples.etf_trend import ETFTrendStrategy
from qmt_agent_trader.strategy.examples.factor_rank_long_only import FactorRankLongOnlyStrategy
from qmt_agent_trader.strategy.models import FactorLeg


def test_factor_rank_long_only_combines_factor_scores() -> None:
    strategy = FactorRankLongOnlyStrategy(
        factors=[FactorLeg(factor_id="momentum", weight=1.0)],
        top_n=1,
        max_single_position_pct=0.5,
    )
    result = strategy.generate_signals(
        pd.DataFrame(
            {
                "symbol": ["A", "B"],
                "trade_date": ["2024-01-02", "2024-01-02"],
                "momentum": [0.1, 0.5],
            }
        )
    )

    assert result["symbol"].tolist() == ["B"]
    assert result.loc[0, "target_weight"] == 0.5


def test_etf_trend_outputs_risk_on_signal() -> None:
    rows = []
    start = date(2024, 1, 1)
    for offset in range(5):
        rows.append(
            {
                "symbol": "ETF",
                "trade_date": start + timedelta(days=offset),
                "close": 10.0 + offset,
            }
        )
    strategy = ETFTrendStrategy(short_ma=2, long_ma=3, max_single_position_pct=1.0)

    result = strategy.generate_signals(pd.DataFrame(rows))

    assert result["symbol"].tolist() == ["ETF"]
    assert result.loc[0, "target_weight"] == 1.0
