from __future__ import annotations

from pathlib import Path

import pytest

from qmt_agent_trader.data.storage import DataLake
from qmt_agent_trader.strategy import execution_adapter
from qmt_agent_trader.strategy.execution_adapter import (
    StrategyBacktestConfig,
    run_strategy_backtest,
)
from qmt_agent_trader.strategy.models import StrategySpec
from qmt_agent_trader.strategy.registry import StrategyRegistry


def _spec(strategy_id: str) -> StrategySpec:
    return StrategySpec.model_validate(
        {
            "strategy_id": strategy_id,
            "name": strategy_id,
            "kind": "FACTOR_RANK_LONG_ONLY",
            "factors": [{"factor_id": "momentum_20d"}],
            "portfolio": {"top_n": 20},
            "rebalance": {"frequency": "daily"},
        }
    )


@pytest.mark.parametrize("identity_mode", ["adhoc", "inline"])
def test_non_registry_identity_never_queries_registry(
    tmp_path,
    monkeypatch,
    identity_mode: str,
) -> None:
    lake = DataLake(tmp_path / "lake", tmp_path / "research.duckdb")
    registry = StrategyRegistry(tmp_path / "strategies")
    spec = _spec(f"{identity_mode}_strategy")
    config = StrategyBacktestConfig(
        strategy_id=spec.strategy_id,
        strategy_identity_mode=identity_mode,
        strategy_spec=spec,
        factor_name="momentum_20d",
        start_date="20240101",
        end_date="20240131",
    )

    monkeypatch.setattr(
        execution_adapter,
        "_strategy_from_registry",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("Registry must not be queried")
        ),
    )
    monkeypatch.setattr(
        execution_adapter,
        "load_session_window",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("identity stage passed")
        ),
    )

    with pytest.raises(RuntimeError, match="identity stage passed"):
        run_strategy_backtest(
            lake,
            registry,
            config,
            reports_dir=Path(tmp_path / "reports"),
        )


def test_registry_identity_requires_saved_strategy(tmp_path) -> None:
    lake = DataLake(tmp_path / "lake", tmp_path / "research.duckdb")
    registry = StrategyRegistry(tmp_path / "strategies")
    spec = _spec("missing_saved_strategy")
    config = StrategyBacktestConfig(
        strategy_id=spec.strategy_id,
        strategy_identity_mode="registry",
        strategy_spec=spec,
        factor_name="momentum_20d",
        start_date="20240101",
        end_date="20240131",
    )

    result = run_strategy_backtest(
        lake,
        registry,
        config,
        reports_dir=Path(tmp_path / "reports"),
    )

    assert result.status == "BLOCKED"
    assert result.reason == "STRATEGY_NOT_FOUND"
