import json

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


def test_strategy_registry_migrates_v1_without_persisting_builtins(tmp_path) -> None:
    legacy_record = _saved("legacy_strategy").model_dump(mode="json")
    legacy = {"version": 1, "strategies": [legacy_record]}
    path = tmp_path / "registry.json"
    path.write_text(json.dumps(legacy), encoding="utf-8")

    registry = StrategyRegistry(tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert registry.get_strategy("legacy_strategy") is not None
    assert payload["schema_version"] == 2
    assert [item["strategy_id"] for item in payload["items"]] == ["legacy_strategy"]
    assert json.loads((tmp_path / "registry.json.v1.bak").read_text()) == legacy


def test_strategy_registry_preserves_trusted_approval_gate_after_migration(tmp_path) -> None:
    registry = StrategyRegistry(tmp_path)
    registry.save_candidate(_saved("strat_review", ApprovalStatus.REVIEW_REQUIRED))

    with pytest.raises(ValueError, match="trusted"):
        registry.update_status("strat_review", ApprovalStatus.APPROVED)

    approved = registry.update_status(
        "strat_review",
        ApprovalStatus.APPROVED,
        trusted=True,
    )
    assert approved.status == ApprovalStatus.APPROVED
