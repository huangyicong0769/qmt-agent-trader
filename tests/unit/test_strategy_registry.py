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


def test_strategy_registry_attaches_generated_implementation_to_agent_draft(tmp_path) -> None:
    registry = StrategyRegistry(tmp_path)
    draft = _saved("strat_test").model_copy(
        update={
            "implementation_ref": "spec:draft",
            "code_path": None,
            "tests_path": None,
        }
    )
    registry.save_candidate(draft)

    updated = registry.attach_generated_implementation(
        "strat_test",
        spec=draft.spec,
        code_path="/tmp/strategy.py",
        tests_path="/tmp/test_strategy.py",
    )

    assert updated.implementation_ref == "file:/tmp/strategy.py"
    assert updated.code_path == "/tmp/strategy.py"
    assert updated.tests_path == "/tmp/test_strategy.py"
    assert registry.get_strategy("strat_test") == updated


def test_strategy_registry_rejects_generated_implementation_for_builtin(tmp_path) -> None:
    registry = StrategyRegistry(tmp_path)

    with pytest.raises(ValueError, match="built-in"):
        registry.attach_generated_implementation(
            "factor_rank_long_only_v1",
            spec=StrategySpec(strategy_id="factor_rank_long_only_v1", name="builtin"),
            code_path="/tmp/strategy.py",
            tests_path=None,
        )


def test_strategy_registry_rejects_direct_approved_candidate(tmp_path) -> None:
    registry = StrategyRegistry(tmp_path)

    with pytest.raises(ValueError, match="APPROVED"):
        registry.save_candidate(_saved("strat_test", ApprovalStatus.APPROVED))
