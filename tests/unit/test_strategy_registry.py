import pytest

from qmt_agent_trader.core.types import ApprovalStatus
from qmt_agent_trader.strategy.models import SavedStrategy, StrategySource, StrategySpec
from qmt_agent_trader.strategy.registry import StrategyRegistry


def _saved(
    strategy_id: str,
    status: ApprovalStatus = ApprovalStatus.GENERATED_BY_LLM,
) -> SavedStrategy:
    spec = StrategySpec(strategy_id=strategy_id, name=strategy_id)
    return SavedStrategy(
        strategy_id=strategy_id,
        name=strategy_id,
        version="0.1.0",
        source=StrategySource.AGENT_GENERATED,
        status=status,
        spec=spec,
        implementation_ref=f"file:{strategy_id}.py",
    )


def test_strategy_registry_saves_reads_and_searches_candidate(tmp_path) -> None:
    registry = StrategyRegistry(tmp_path)
    registry.save_candidate(_saved("strat_test"))

    assert registry.get_strategy("strat_test") is not None
    assert registry.find_strategies("test")[0].strategy_id == "strat_test"


def test_strategy_registry_rejects_duplicate_id(tmp_path) -> None:
    registry = StrategyRegistry(tmp_path)
    registry.save_candidate(_saved("strat_test"))

    with pytest.raises(ValueError, match="already registered"):
        registry.save_candidate(_saved("strat_test"))


def test_strategy_registry_rejects_direct_approved_candidate(tmp_path) -> None:
    registry = StrategyRegistry(tmp_path)

    with pytest.raises(ValueError, match="APPROVED"):
        registry.save_candidate(_saved("strat_test", ApprovalStatus.APPROVED))
