import pytest
from pydantic import ValidationError

from qmt_agent_trader.strategy.models import StrategyKind, strategy_spec_from_agent_spec


def test_strategy_spec_converts_agent_shape() -> None:
    spec = strategy_spec_from_agent_spec(
        {
            "strategy_id": "strat_1",
            "name": "Momentum",
            "factors": ["momentum_20d"],
            "portfolio_construction": {"method": "equal_weight", "top_n": 5},
            "execution_assumptions": {"timing": "next_open", "slippage_model": "fixed_5bps"},
        }
    )

    assert spec.strategy_id == "strat_1"
    assert spec.kind == StrategyKind.FACTOR_RANK_LONG_ONLY
    assert spec.factors[0].factor_id == "momentum_20d"
    assert spec.portfolio.top_n == 5
    assert spec.execution.execution_timing == "next_open"
    assert spec.execution.slippage_bps == 5.0


def test_strategy_spec_rejects_factor_name_as_factor_leg_alias() -> None:
    with pytest.raises(ValidationError):
        strategy_spec_from_agent_spec(
            {
                "strategy_id": "strat_factor_name",
                "name": "Bad alias",
                "factors": [{"factor_name": "volatility_20d", "weight": 1.0}],
            }
        )
