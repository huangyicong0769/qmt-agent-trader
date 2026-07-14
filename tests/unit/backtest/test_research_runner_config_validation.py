from datetime import date

import pytest
from pydantic import ValidationError

from qmt_agent_trader.backtest.research_runner import FactorRankResearchConfig
from qmt_agent_trader.strategy.execution_adapter import StrategyBacktestConfig


@pytest.mark.parametrize(
    ("update", "field"),
    [
        ({"initial_cash": 0}, "initial_cash"),
        ({"execution_delay_days": 0}, "execution_delay_days"),
        ({"slippage_bps": -1}, "slippage_bps"),
        ({"top_n": 0}, "top_n"),
        ({"max_single_position_pct": 1.1}, "max_single_position_pct"),
    ],
)
def test_strategy_backtest_config_rejects_invalid_numeric_values(update, field) -> None:
    payload = {
        "strategy_id": "factor_rank",
        "start_date": "20240101",
        "end_date": "20240131",
        **update,
    }

    with pytest.raises(ValidationError) as exc_info:
        StrategyBacktestConfig.model_validate(payload)

    assert field in str(exc_info.value)


@pytest.mark.parametrize(
    ("update", "message"),
    [
        ({"initial_cash": float("inf")}, "initial_cash must be finite and positive"),
        ({"initial_cash": 0}, "initial_cash must be finite and positive"),
        ({"top_n": 0}, "top_n must be positive"),
        ({"max_single_position_pct": 0}, "max_single_position_pct must be in (0, 1]"),
        ({"cash_buffer_pct": 1}, "cash_buffer_pct must be in [0, 1)"),
        ({"min_turnover_threshold": -0.1}, "min_turnover_threshold must be in [0, 1]"),
        ({"rank_buffer": -1}, "rank_buffer must be non-negative"),
        ({"rebalance_frequency": "yearly"}, "unsupported rebalance_frequency: yearly"),
        ({"expected_trade_dates": ()}, "expected_trade_dates cannot be empty"),
        (
            {
                "expected_trade_dates": (
                    date(2024, 1, 3),
                    date(2024, 1, 2),
                )
            },
            "expected_trade_dates must be sorted and unique",
        ),
        (
            {
                "expected_trade_dates": (
                    date(2024, 1, 2),
                    date(2024, 1, 2),
                )
            },
            "expected_trade_dates must be sorted and unique",
        ),
    ],
)
def test_research_config_rejects_invalid_values(update, message) -> None:
    payload = {
        "factor_name": "fixture",
        "expected_trade_dates": (date(2024, 1, 2),),
        **update,
    }

    with pytest.raises(ValueError) as exc_info:
        FactorRankResearchConfig(**payload)

    assert str(exc_info.value) == message
