from qmt_agent_trader.agent.tools import strategy_tools
from qmt_agent_trader.data.storage import DataLake
from qmt_agent_trader.strategy.execution_adapter import StrategyBacktestConfig


def test_strategy_backtest_config_serializes_declared_semantics() -> None:
    config = StrategyBacktestConfig(
        strategy_id="weekly_low_value",
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


def test_backtest_cache_key_changes_with_execution_semantics(tmp_path, monkeypatch) -> None:
    lake = DataLake(tmp_path, tmp_path / "research.duckdb")
    monkeypatch.setattr(strategy_tools, "_data_fingerprint", lambda _lake: "data")
    monkeypatch.setattr(
        strategy_tools,
        "_factor_fingerprint",
        lambda _lake, _factor_ids: "factor",
    )
    base = StrategyBacktestConfig(
        strategy_id="strategy",
        start_date="20240101",
        end_date="20240630",
    )
    weekly = base.model_copy(update={"rebalance_frequency": "weekly"})
    buffered = base.model_copy(update={"cash_buffer_pct": 0.10})

    def key(config: StrategyBacktestConfig) -> str:
        return strategy_tools._backtest_cache_key(
            lake,
            config=config,
            factor_name="momentum_20d",
            requested_factor_ids=["momentum_20d"],
        )

    assert key(base) != key(weekly)
    assert key(base) != key(buffered)
