from qmt_agent_trader.agent.tools import strategy_tools
from qmt_agent_trader.strategy.execution_adapter import StrategyBacktestConfig


def test_strategy_backtest_config_serializes_declared_semantics() -> None:
    config = StrategyBacktestConfig(
        strategy_id="weekly_low_value",
        strategy_identity_mode="adhoc",
        start_date="20240101",
        end_date="20240630",
        rebalance_frequency="weekly",
        min_turnover_threshold=0.08,
        rank_buffer=20,
        cash_buffer_pct=0.10,
        lower_is_better=True,
    )

    payload = config.model_dump(mode="json")
    assert payload["rebalance_frequency"] == "weekly"
    assert payload["min_turnover_threshold"] == 0.08
    assert payload["rank_buffer"] == 20
    assert payload["cash_buffer_pct"] == 0.10
    assert payload["lower_is_better"] is True


def test_backtest_cache_key_changes_with_execution_semantics() -> None:
    base = StrategyBacktestConfig(
        strategy_id="strategy",
        strategy_identity_mode="adhoc",
        start_date="20240101",
        end_date="20240630",
    )
    weekly = base.model_copy(update={"rebalance_frequency": "weekly"})
    buffered = base.model_copy(update={"cash_buffer_pct": 0.10})

    def key(config: StrategyBacktestConfig) -> str:
        return strategy_tools._backtest_cache_key(
            config=config,
            factor_name="momentum_20d",
            requested_factor_ids=["momentum_20d"],
            provenance={"dataset_fingerprints": {"tushare/daily": "stable"}},
        )

    assert key(base) != key(weekly)
    assert key(base) != key(buffered)
