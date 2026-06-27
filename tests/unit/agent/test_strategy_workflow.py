"""Tests for the Strategy Engineering workflow."""

from __future__ import annotations

import pytest

from qmt_agent_trader.agent.experiment_store import ExperimentStore
from qmt_agent_trader.agent.sandbox import CodeSandbox
from qmt_agent_trader.agent.schemas import ExperimentStatus, ToolContext
from qmt_agent_trader.agent.tools import build_agent_registry
from qmt_agent_trader.agent.workflows.strategy_engineering import (
    StrategyEngineeringWorkflow,
)
from qmt_agent_trader.data.storage import DataLake


@pytest.fixture
def lake(tmp_path):
    return DataLake(
        root=tmp_path / "lake",
        duckdb_path=tmp_path / "test.duckdb",
    )


@pytest.fixture
def registry(lake, tmp_path):
    reg = build_agent_registry(
        data_lake=lake,
        audit_path=tmp_path / "audit.jsonl",
        experiment_root=tmp_path / "experiments",
        sandbox=CodeSandbox(tmp_path / "generated"),
    )
    import pandas as pd

    lake.write_parquet(
        pd.DataFrame(
            {
                "ts_code": ["000001.SZ"],
                "trade_date": ["20240102"],
                "open": [10],
                "high": [11],
                "low": [9],
                "close": [10.5],
                "vol": [1000000],
                "amount": [10000000],
            }
        ),
        "raw",
        "tushare_daily",
    )
    return reg


@pytest.fixture
def store(tmp_path):
    return ExperimentStore(tmp_path / "experiments")


def test_strategy_workflow_runs(registry, store):
    wf = StrategyEngineeringWorkflow(registry, store)
    exp = wf.run(
        "基于动量的日频轮动策略",
        ["momentum_20d"],
        "stock_etf",
        "20200101",
        "20240624",
    )
    assert exp.experiment_id is not None
    assert exp.kind == "strategy_engineering"


def test_strategy_workflow_creates_experiment(registry, store):
    wf = StrategyEngineeringWorkflow(registry, store)
    exp = wf.run("test", ["reversal_5d"], "stock", "20200101", "20240624")
    assert exp.status in (ExperimentStatus.REVIEW_REQUIRED, ExperimentStatus.FAILED)


def test_strategy_workflow_does_not_call_broker(registry, store):
    """Ensure the strategy workflow cannot generate live broker interactions."""
    wf = StrategyEngineeringWorkflow(registry, store)
    exp = wf.run("safe test", ["momentum_20d"], "stock", "20200101", "20240624")
    # Verify the experiment artifacts contain no broker references
    artifacts = [a.lower() for a in exp.artifacts]
    assert not any("broker" in a for a in artifacts)
    assert not any("gateway" in a for a in artifacts)
    assert not any("submit_order" in a for a in artifacts)


def test_agent_can_list_generated_strategy_candidates(registry):
    context = ToolContext(run_id="strategy-list")
    spec = registry.run_tool(
        "create_strategy_spec",
        {
            "strategy_idea": "基于动量的候选策略",
            "selected_factors": ["momentum_20d"],
        },
        context,
    )["strategy_spec"]
    generated = registry.run_tool("generate_strategy_code", {"strategy_spec": spec}, context)

    listed = registry.run_tool(
        "list_strategy_candidates",
        {"query": spec["strategy_id"]},
        context,
    )

    assert generated["status"] == "generated"
    assert listed["status"] == "ok"
    assert listed["count"] == 1
    assert listed["strategies"][0]["strategy_id"] == spec["strategy_id"]
    assert listed["strategies"][0]["status"] == "draft"


def test_agent_can_list_legacy_flat_strategy_candidates(registry, tmp_path):
    context = ToolContext(run_id="strategy-list-flat")
    generated = tmp_path / "generated" / "strategies" / "strat_legacy.py"
    generated.parent.mkdir(parents=True, exist_ok=True)
    generated.write_text("def generate_signals(data):\n    return data\n", encoding="utf-8")
    generated.with_name("test_strat_legacy.py").write_text(
        "def test_placeholder():\n    assert True\n",
        encoding="utf-8",
    )

    listed = registry.run_tool(
        "list_strategy_candidates",
        {"query": "strat_legacy"},
        context,
    )

    assert listed["status"] == "ok"
    assert listed["count"] == 1
    assert listed["strategies"][0]["strategy_id"] == "strat_legacy"
    assert listed["strategies"][0]["tests_path"].endswith("test_strat_legacy.py")
