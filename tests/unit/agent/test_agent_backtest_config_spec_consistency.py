from qmt_agent_trader.agent.schemas import ToolContext
from qmt_agent_trader.agent.tools import strategy_tools
from qmt_agent_trader.data.storage import DataLake
from qmt_agent_trader.strategy.models import StrategySpec


def _spec() -> StrategySpec:
    return StrategySpec.model_validate(
        {
            "strategy_id": "weekly_value",
            "name": "Weekly value",
            "kind": "FACTOR_RANK_LONG_ONLY",
            "factors": [{"factor_id": "pb", "ascending": True}],
            "portfolio": {"top_n": 10},
            "rebalance": {"frequency": "weekly"},
            "execution": {"execution_delay_days": 1},
        }
    )


def test_agent_blocks_config_spec_mismatch_before_data_access(tmp_path, monkeypatch) -> None:
    lake = DataLake(tmp_path / "lake", tmp_path / "research.duckdb")
    monkeypatch.setattr(strategy_tools, "_get_lake", lambda: lake)
    monkeypatch.setattr(
        strategy_tools,
        "_resolve_backtest_universe",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("universe data access must not run")
        ),
    )

    result = strategy_tools._run_backtest(
        {
            "strategy_spec": _spec().model_dump(mode="json"),
            "factor_name": "pb",
            "top_n": 20,
            "start_date": "20240101",
            "end_date": "20240630",
        },
        ToolContext(run_id="config-spec-mismatch"),
    )

    assert result["status"] == "BLOCKED"
    assert result["reason"] == "CONFIG_SPEC_MISMATCH"
    assert result["unsupported_fields"] == ["config.top_n"]


def test_strategy_rebalance_frequency_mismatch_blocks_before_universe_access(
    tmp_path,
    monkeypatch,
) -> None:
    lake = DataLake(tmp_path / "lake", tmp_path / "research.duckdb")
    monkeypatch.setattr(strategy_tools, "_get_lake", lambda: lake)
    monkeypatch.setattr(
        strategy_tools,
        "_resolve_backtest_universe",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("mismatch must block before universe resolution")
        ),
    )

    result = strategy_tools._run_backtest(
        {
            "strategy_spec": _spec().model_dump(mode="json"),
            "rebalance_frequency": "daily",
            "start_date": "20240101",
            "end_date": "20240630",
        },
        ToolContext(run_id="frequency-mismatch"),
    )

    assert result["status"] == "BLOCKED"
    assert result["reason"] == "CONFIG_SPEC_MISMATCH"
    assert result["unsupported_fields"] == ["config.rebalance_frequency"]
