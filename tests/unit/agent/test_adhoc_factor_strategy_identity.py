from __future__ import annotations

import pytest

from qmt_agent_trader.agent.tools import strategy_tools
from qmt_agent_trader.core.types import ApprovalStatus
from qmt_agent_trader.data.storage import DataLake
from qmt_agent_trader.strategy.models import SavedStrategy, StrategySource, StrategySpec


@pytest.fixture
def wired_strategy_tools(tmp_path, monkeypatch) -> DataLake:
    lake = DataLake(tmp_path / "lake", tmp_path / "research.duckdb")
    monkeypatch.setattr(strategy_tools, "_get_lake", lambda: lake)
    return lake


def test_factor_only_request_does_not_load_same_named_saved_strategy(
    wired_strategy_tools,
) -> None:
    saved_spec = StrategySpec.model_validate(
        {
            "strategy_id": "adhoc_factor_momentum_20d",
            "name": "Saved collision",
            "kind": "FACTOR_RANK_LONG_ONLY",
            "factors": [{"factor_id": "momentum_20d"}],
            "portfolio": {"top_n": 3},
            "rebalance": {"frequency": "monthly"},
        }
    )
    strategy_tools._strategy_registry().save_candidate(
        SavedStrategy(
            strategy_id=saved_spec.strategy_id,
            name=saved_spec.name,
            version=saved_spec.version,
            source=StrategySource.AGENT_GENERATED,
            status=ApprovalStatus.DRAFT,
            spec=saved_spec,
            implementation_ref="spec:draft",
        )
    )

    intent = strategy_tools._resolve_backtest_intent(
        {"factor_name": "momentum_20d"},
        requested_strategy_frequency="weekly",
        requested_top_n=20,
    )

    assert not isinstance(intent, dict)
    assert intent.saved_strategy is None
    assert intent.strategy_id == "adhoc_factor_momentum_20d"
    assert intent.strategy_identity_mode == "adhoc"
    assert intent.strategy_spec.portfolio.top_n == 20
    assert intent.strategy_spec.rebalance.frequency == "weekly"
