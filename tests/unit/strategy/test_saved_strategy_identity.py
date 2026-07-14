from pathlib import Path

from qmt_agent_trader.core.types import ApprovalStatus
from qmt_agent_trader.data.storage import DataLake
from qmt_agent_trader.strategy.execution_adapter import (
    StrategyBacktestConfig,
    run_strategy_backtest,
)
from qmt_agent_trader.strategy.models import SavedStrategy, StrategySource, StrategySpec
from qmt_agent_trader.strategy.registry import StrategyRegistry


def _spec(strategy_id: str = "saved_value", *, top_n: int = 10) -> StrategySpec:
    return StrategySpec.model_validate(
        {
            "strategy_id": strategy_id,
            "name": "Saved value",
            "kind": "FACTOR_RANK_LONG_ONLY",
            "factors": [{"factor_id": "pb_rank", "ascending": True}],
            "portfolio": {"top_n": top_n},
            "rebalance": {"frequency": "weekly"},
            "execution": {"execution_delay_days": 1},
        }
    )


def _save(registry: StrategyRegistry, spec: StrategySpec) -> None:
    registry.save_candidate(
        SavedStrategy(
            strategy_id=spec.strategy_id,
            name=spec.name,
            version=spec.version,
            source=StrategySource.AGENT_GENERATED,
            status=ApprovalStatus.DRAFT,
            spec=spec,
            implementation_ref="spec:draft",
            code_path=None,
        )
    )


def test_config_strategy_id_must_equal_inline_spec_id(tmp_path) -> None:
    lake = DataLake(tmp_path / "lake", tmp_path / "research.duckdb")
    config = StrategyBacktestConfig(
        strategy_id="different_id",
        strategy_spec=_spec(),
        factor_name="pb_rank",
        start_date="20240101",
        end_date="20240331",
        top_n=10,
        rebalance_frequency="weekly",
        lower_is_better=True,
    )

    result = run_strategy_backtest(
        lake,
        StrategyRegistry(tmp_path / "strategies"),
        config,
        reports_dir=Path(tmp_path / "reports"),
    )

    assert result.status == "BLOCKED"
    assert result.reason == "CONFIG_SPEC_MISMATCH"
    assert result.unsupported_fields == ["config.strategy_id"]


def test_inline_spec_cannot_replace_saved_registry_spec(tmp_path) -> None:
    lake = DataLake(tmp_path / "lake", tmp_path / "research.duckdb")
    registry = StrategyRegistry(tmp_path / "strategies")
    _save(registry, _spec(top_n=10))
    config = StrategyBacktestConfig(
        strategy_id="saved_value",
        strategy_spec=_spec(top_n=20),
        factor_name="pb_rank",
        start_date="20240101",
        end_date="20240331",
        top_n=20,
        rebalance_frequency="weekly",
        lower_is_better=True,
    )

    result = run_strategy_backtest(
        lake,
        registry,
        config,
        reports_dir=Path(tmp_path / "reports"),
    )

    assert result.status == "BLOCKED"
    assert result.reason == "SAVED_STRATEGY_SPEC_MISMATCH"
