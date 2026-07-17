from __future__ import annotations

from pathlib import Path

import pytest

from qmt_agent_trader.agent.sandbox import CodeSandbox
from qmt_agent_trader.agent.schemas import ToolContext
from qmt_agent_trader.agent.tools import build_agent_registry
from qmt_agent_trader.data.storage import DataLake


@pytest.fixture
def registry(tmp_path: Path):
    lake = DataLake(root=tmp_path / "lake", duckdb_path=tmp_path / "test.duckdb")
    return build_agent_registry(
        data_lake=lake,
        audit_path=tmp_path / "audit.jsonl",
        experiment_root=tmp_path / "experiments",
        sandbox=CodeSandbox(tmp_path / "generated"),
    )


def test_strategy_spec_preserves_weights_directions_and_constraints(registry) -> None:
    result = registry.run_tool(
        "create_strategy_spec",
        {
            "strategy_idea": "weighted factor strategy",
            "selected_factors": [
                {"factor_id": "momentum_20d", "weight": 0.5, "ascending": False},
                {"factor_id": "volatility_20d", "weight": 0.3, "ascending": True},
                "turnover_20d",
            ],
            "constraints": {
                "factor_weights": {"turnover_20d": 0.2},
                "factor_directions": {"turnover_20d": "lower_is_better"},
                "top_n": 12,
                "max_single_position_pct": 0.08,
                "cash_buffer_pct": 0.03,
                "execution_delay_days": 2,
                "slippage_bps": 7,
            },
        },
        ToolContext(run_id="strategy-spec-fidelity"),
    )

    spec = result["strategy_spec"]
    assert result["saved_in_registry"] is False
    assert result["research_only"] is True
    assert result["live_trading_allowed"] is False
    assert "save_strategy_spec_draft" in result["suggested_next_tools"]
    assert spec["factors"] == [
        {"factor_id": "momentum_20d", "weight": 0.5, "ascending": False, "transform": None},
        {"factor_id": "volatility_20d", "weight": 0.3, "ascending": True, "transform": None},
        {"factor_id": "turnover_20d", "weight": 0.2, "ascending": True, "transform": None},
    ]
    assert spec["portfolio"]["top_n"] == 12
    assert spec["portfolio"]["max_single_position_pct"] == 0.08
    assert spec["portfolio"]["cash_buffer_pct"] == 0.03
    assert spec["execution"]["execution_delay_days"] == 2
    assert spec["execution"]["slippage_bps"] == 7.0
    assert spec["risk_constraints"] == {}


def test_strategy_spec_draft_can_be_saved_and_resolved_by_strategy_id(registry) -> None:
    context = ToolContext(run_id="strategy-spec-draft")
    spec = registry.run_tool(
        "create_strategy_spec",
        {
            "strategy_idea": "draft strategy",
            "selected_factors": ["momentum_20d"],
            "universe": "stock_etf",
        },
        context,
    )["strategy_spec"]

    saved = registry.run_tool("save_strategy_spec_draft", {"strategy_spec": spec}, context)
    result = registry.run_tool(
        "run_backtest",
        {
            "strategy_id": spec["strategy_id"],
            "start_date": "20240101",
            "end_date": "20240102",
            "symbols": ["000001.SZ"],
        },
        context,
    )

    assert saved["status"] == "saved"
    assert saved["code_path"] is None
    assert saved["saved_in_registry"] is True
    assert result["status"] != "STRATEGY_NOT_FOUND"


def test_strategy_spec_draft_can_be_upgraded_with_generated_code(registry) -> None:
    context = ToolContext(run_id="strategy-spec-draft-upgrade")
    spec = registry.run_tool(
        "create_strategy_spec",
        {
            "strategy_idea": "draft strategy with generated implementation",
            "selected_factors": [
                {"factor_id": "momentum_20d", "weight": 0.6, "ascending": False},
                {"factor_id": "volatility_20d", "weight": 0.4, "ascending": True},
            ],
        },
        context,
    )["strategy_spec"]
    registry.run_tool("save_strategy_spec_draft", {"strategy_spec": spec}, context)
    generated = registry.run_tool("generate_strategy_code", {"strategy_spec": spec}, context)

    saved = registry.run_tool(
        "save_strategy_candidate",
        {
            "strategy_spec": spec,
            "code_path": generated["code_path"],
            "tests_path": generated["tests_path"],
        },
        context,
    )

    assert saved["status"] == "updated"
    assert saved["registry_action"] == "attached_generated_implementation"
    assert saved["saved_strategy"]["implementation_ref"] == f"file:{generated['code_path']}"
    assert saved["saved_strategy"]["code_path"] == generated["code_path"]
    assert saved["saved_strategy"]["tests_path"] == generated["tests_path"]
    assert saved["review_required"] is True
    assert saved["live_trading_allowed"] is False


def test_unknown_strategy_id_returns_structured_error(registry) -> None:
    result = registry.run_tool(
        "run_backtest",
        {"strategy_id": "strat_missing"},
        ToolContext(run_id="strategy-id-missing"),
    )

    assert result["status"] == "STRATEGY_NOT_FOUND"
    assert result["strategy_id"] == "strat_missing"
    assert "save_strategy_spec_draft" in result["suggested_next_tools"]
