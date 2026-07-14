from pathlib import Path

import pytest

from qmt_agent_trader.data.storage import DataLake
from qmt_agent_trader.strategy.execution_adapter import (
    StrategyBacktestConfig,
    run_strategy_backtest,
    validate_backtest_config_matches_spec,
)
from qmt_agent_trader.strategy.models import StrategySpec
from qmt_agent_trader.strategy.registry import StrategyRegistry


def spec() -> StrategySpec:
    return StrategySpec.model_validate(
        {
            "strategy_id": "weekly_value",
            "name": "Weekly value",
            "kind": "FACTOR_RANK_LONG_ONLY",
            "factors": [{"factor_id": "pb", "ascending": True}],
            "portfolio": {
                "method": "equal_weight_top_n",
                "top_n": 10,
                "max_single_position_pct": 0.08,
                "cash_buffer_pct": 0.10,
                "long_only": True,
            },
            "rebalance": {
                "frequency": "weekly",
                "min_turnover_threshold": 0.05,
                "rank_buffer": 10,
            },
            "execution": {
                "signal_timing": "after_close",
                "execution_timing": "next_open",
                "execution_delay_days": 1,
                "slippage_bps": 5.0,
                "cost_model": "a_share_default",
            },
        }
    )


def matching_config() -> StrategyBacktestConfig:
    return StrategyBacktestConfig(
        strategy_id="weekly_value",
        strategy_spec=spec(),
        factor_name="pb",
        start_date="20240101",
        end_date="20240630",
        top_n=10,
        max_single_position_pct=0.08,
        cash_buffer_pct=0.10,
        rebalance_frequency="weekly",
        min_turnover_threshold=0.05,
        rank_buffer=10,
        execution_delay_days=1,
        slippage_bps=5.0,
        lower_is_better=True,
    )


@pytest.mark.parametrize(
    ("update", "field"),
    [
        ({"factor_name": "momentum_20d"}, "config.factor_name"),
        ({"top_n": 20}, "config.top_n"),
        ({"max_single_position_pct": 0.10}, "config.max_single_position_pct"),
        ({"cash_buffer_pct": 0.02}, "config.cash_buffer_pct"),
        ({"rebalance_frequency": "daily"}, "config.rebalance_frequency"),
        ({"min_turnover_threshold": 0.0}, "config.min_turnover_threshold"),
        ({"rank_buffer": 0}, "config.rank_buffer"),
        ({"execution_delay_days": 2}, "config.execution_delay_days"),
        ({"slippage_bps": 10.0}, "config.slippage_bps"),
        ({"lower_is_better": False}, "config.lower_is_better"),
    ],
)
def test_config_cannot_override_spec(update, field) -> None:
    config = matching_config().model_copy(update=update)
    issues = validate_backtest_config_matches_spec(config, spec())
    assert field in {issue.field for issue in issues}


def test_matching_config_has_no_mismatch() -> None:
    assert validate_backtest_config_matches_spec(matching_config(), spec()) == ()


def test_direct_adapter_blocks_mismatch_before_data_access(tmp_path, monkeypatch) -> None:
    lake = DataLake(tmp_path / "lake", tmp_path / "research.duckdb")
    monkeypatch.setattr(
        "qmt_agent_trader.strategy.execution_adapter.build_target_frequency_panel",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("data access must not run")
        ),
    )

    result = run_strategy_backtest(
        lake,
        StrategyRegistry(tmp_path / "strategies"),
        matching_config().model_copy(update={"top_n": 20}),
        reports_dir=Path(tmp_path / "reports"),
    )

    assert result.status == "BLOCKED"
    assert result.reason == "CONFIG_SPEC_MISMATCH"
    assert result.unsupported_fields == ["config.top_n"]
