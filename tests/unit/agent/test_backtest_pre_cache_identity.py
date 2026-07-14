from __future__ import annotations

import pytest

from qmt_agent_trader.agent.schemas import ToolContext
from qmt_agent_trader.agent.tools import strategy_tools
from qmt_agent_trader.core.types import ApprovalStatus
from qmt_agent_trader.data.storage import DataLake
from qmt_agent_trader.strategy.models import (
    SavedStrategy,
    StrategyKind,
    StrategySource,
    StrategySpec,
)
from qmt_agent_trader.strategy.registry import StrategyRegistry


@pytest.fixture
def wired_lake(tmp_path, monkeypatch) -> DataLake:
    lake = DataLake(tmp_path / "lake", tmp_path / "research.duckdb")
    monkeypatch.setattr(strategy_tools, "_get_lake", lambda: lake)
    monkeypatch.setattr(
        strategy_tools,
        "_resolve_backtest_universe",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("identity guards must finish before universe resolution")
        ),
    )
    monkeypatch.setattr(
        strategy_tools,
        "_get_cached_backtest",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("identity guards must finish before cache lookup")
        ),
    )
    return lake


def test_inline_saved_identity_conflict_blocks_before_universe_or_cache(wired_lake) -> None:
    saved = _save_strategy(wired_lake, strategy_id="inline_saved", code_path=None, top_n=10)
    conflicting = saved.spec.model_copy(
        update={"portfolio": saved.spec.portfolio.model_copy(update={"top_n": 20})}
    )

    result = strategy_tools._run_backtest(
        {
            "strategy_spec": conflicting.model_dump(mode="json"),
            "start_date": "20240101",
            "end_date": "20240331",
            "symbols": ["000001.SZ"],
        },
        ToolContext(run_id="inline-saved-conflict"),
    )

    assert result["status"] == "BLOCKED"
    assert result["reason"] == "SAVED_STRATEGY_SPEC_MISMATCH"


def test_inline_generated_identity_blocks_before_universe_or_cache(wired_lake) -> None:
    saved = _save_strategy(
        wired_lake,
        strategy_id="generated_inline",
        code_path="generated/strategy.py",
    )

    result = strategy_tools._run_backtest(
        {
            "strategy_spec": saved.spec.model_dump(mode="json"),
            "start_date": "20240101",
            "end_date": "20240331",
            "symbols": ["000001.SZ"],
        },
        ToolContext(run_id="generated-pre-cache"),
    )

    assert result["status"] == "BLOCKED"
    assert result["reason"] == "GENERATED_STRATEGY_EXECUTION_NOT_IMPLEMENTED"


def _save_strategy(
    lake: DataLake,
    *,
    strategy_id: str,
    code_path: str | None,
    top_n: int = 20,
) -> SavedStrategy:
    spec = StrategySpec(
        strategy_id=strategy_id,
        name=strategy_id,
        kind=StrategyKind.FACTOR_RANK_LONG_ONLY,
        factors=[{"factor_id": "momentum_20d"}],
        portfolio={"top_n": top_n},
    )
    saved = SavedStrategy(
        strategy_id=strategy_id,
        name=strategy_id,
        version=spec.version,
        source=StrategySource.AGENT_GENERATED,
        status=(ApprovalStatus.GENERATED_BY_LLM if code_path else ApprovalStatus.DRAFT),
        spec=spec,
        implementation_ref=f"file:{code_path}" if code_path else "spec:draft",
        code_path=code_path,
    )
    return StrategyRegistry(lake.root.parent / "strategies").save_candidate(saved)
