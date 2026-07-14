from __future__ import annotations

from pathlib import Path

import pytest

from qmt_agent_trader.agent.schemas import ToolContext
from qmt_agent_trader.agent.tools import strategy_tools
from qmt_agent_trader.core.types import ApprovalStatus
from qmt_agent_trader.data.storage import DataLake
from qmt_agent_trader.strategy.execution_adapter import (
    StrategyBacktestConfig,
    StrategyBacktestResult,
    run_strategy_backtest,
)
from qmt_agent_trader.strategy.models import (
    SavedStrategy,
    StrategyKind,
    StrategySource,
    StrategySpec,
)
from qmt_agent_trader.strategy.registry import StrategyRegistry


@pytest.fixture
def wired_strategy_tools(tmp_path, monkeypatch) -> DataLake:
    lake = DataLake(root=tmp_path / "lake", duckdb_path=tmp_path / "test.duckdb")
    monkeypatch.setattr(strategy_tools, "_get_lake", lambda: lake)
    return lake


def _saved_strategy(
    lake: DataLake,
    *,
    strategy_id: str,
    code_path: str | None,
) -> SavedStrategy:
    spec = StrategySpec(
        strategy_id=strategy_id,
        name=strategy_id,
        kind=StrategyKind.FACTOR_RANK_LONG_ONLY,
        factors=[{"factor_id": "momentum_20d"}],
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


def _input(strategy_id: str) -> dict[str, object]:
    return {
        "strategy_id": strategy_id,
        "symbols": ["000001.SZ", "000002.SZ"],
        "start_date": "20240101",
        "end_date": "20240331",
    }


def test_saved_generated_strategy_is_not_silently_run_by_canonical_adapter(
    wired_strategy_tools,
) -> None:
    saved = _saved_strategy(
        wired_strategy_tools,
        strategy_id="saved_generated",
        code_path="generated/strategy.py",
    )

    result = strategy_tools._run_backtest(
        _input(saved.strategy_id),
        ToolContext(run_id="saved-generated-guard"),
    )

    assert saved.code_path
    assert result["status"] == "BLOCKED"
    assert result["reason"] == "GENERATED_STRATEGY_EXECUTION_NOT_IMPLEMENTED"
    assert result["unsupported_fields"] == ["code_path"]


def test_saved_spec_draft_without_code_path_can_use_canonical_adapter(
    wired_strategy_tools,
    monkeypatch,
) -> None:
    saved = _saved_strategy(
        wired_strategy_tools,
        strategy_id="saved_spec_draft",
        code_path=None,
    )
    monkeypatch.setattr(
        strategy_tools,
        "run_strategy_backtest",
        lambda *_args, **_kwargs: StrategyBacktestResult(
            run_id="research_complete",
            strategy_id=saved.strategy_id,
            strategy_version=saved.version,
            status="completed",
        ),
    )

    result = strategy_tools._run_backtest(
        _input(saved.strategy_id),
        ToolContext(run_id="saved-draft-allowed"),
    )

    assert result["status"] == "completed"


def test_direct_adapter_call_blocks_registry_generated_implementation(
    wired_strategy_tools,
) -> None:
    saved = _saved_strategy(
        wired_strategy_tools,
        strategy_id="direct_saved_generated",
        code_path="generated/direct.py",
    )

    result = run_strategy_backtest(
        wired_strategy_tools,
        StrategyRegistry(wired_strategy_tools.root.parent / "strategies"),
        StrategyBacktestConfig(
            strategy_id=saved.strategy_id,
            start_date="20240101",
            end_date="20240331",
        ),
        reports_dir=Path(wired_strategy_tools.root.parent / "reports"),
    )

    assert result.status == "BLOCKED"
    assert result.reason == "GENERATED_STRATEGY_EXECUTION_NOT_IMPLEMENTED"
    assert result.unsupported_fields == ["code_path"]
