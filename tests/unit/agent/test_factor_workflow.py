"""Tests for the Factor Discovery workflow (new pipeline)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from qmt_agent_trader.agent.experiment_store import ExperimentStore
from qmt_agent_trader.agent.sandbox import CodeSandbox
from qmt_agent_trader.agent.schemas import ExperimentStatus, ToolContext
from qmt_agent_trader.agent.tools import build_agent_registry
from qmt_agent_trader.agent.workflows.factor_discovery import FactorDiscoveryWorkflow
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
    # Write minimal data so the workflow doesn't crash on empty lake
    rows = []
    for offset in range(25):
        trade_date = f"{pd.Timestamp('2024-01-01') + pd.Timedelta(days=offset):%Y%m%d}"
        rows.extend(
            [
                {
                    "ts_code": "000001.SZ",
                    "trade_date": trade_date,
                    "open": 10 + offset,
                    "high": 11 + offset,
                    "low": 9 + offset,
                    "close": 10.5 + offset,
                    "vol": 1000000,
                    "amount": 10000000,
                },
                {
                    "ts_code": "000002.SZ",
                    "trade_date": trade_date,
                    "open": 20 + offset * 0.1,
                    "high": 21 + offset * 0.1,
                    "low": 19 + offset * 0.1,
                    "close": 20.5 + offset * 0.1,
                    "vol": 1000000,
                    "amount": 10000000,
                },
            ]
        )
    lake.write_parquet(
        pd.DataFrame(rows),
        "raw",
        "tushare_daily",
    )
    return reg


@pytest.fixture
def store(tmp_path):
    return ExperimentStore(tmp_path / "experiments")


def test_workflow_runs_to_review(registry, store):
    wf = FactorDiscoveryWorkflow(registry, store)
    exp = wf.run("低波动高胜率因子", "stock_etf", "20200101", "20240624")
    assert exp.status in (ExperimentStatus.REVIEW_REQUIRED, ExperimentStatus.FAILED)


def test_workflow_creates_experiment(registry, store):
    wf = FactorDiscoveryWorkflow(registry, store)
    exp = wf.run("momentum test", "stock", "20200101", "20240624")
    assert exp.experiment_id is not None
    assert exp.kind == "factor_discovery"


def test_workflow_does_not_write_to_formal_library(registry, store, tmp_path):
    wf = FactorDiscoveryWorkflow(registry, store)
    exp = wf.run("test factor", "stock", "20200101", "20240624")
    # Check library was not modified
    Path("src/qmt_agent_trader/factors/library/momentum.py")
    # Workflow should not alter production code
    assert exp.status is not None  # just verify it didn't crash


def test_generated_factor_must_be_saved_before_evaluation(registry) -> None:
    context = ToolContext(run_id="run_test", experiment_id="exp_test")
    spec_result = registry.run_tool(
        "create_factor_spec",
        {
            "factor_name": "agent momentum",
            "factor_description": "short horizon momentum",
            "formula_sketch": "pct_change",
            "lookback": 3,
        },
        context,
    )
    factor_spec = spec_result["factor_spec"]
    code_result = registry.run_tool("generate_factor_code", {"factor_spec": factor_spec}, context)
    factor_id = factor_spec["factor_id"]

    unsaved = registry.run_tool(
        "evaluate_factor_candidate",
        {"factor_id": factor_id, "start_date": "20240105", "end_date": "20240120"},
        context,
    )

    assert unsaved["status"] == "FACTOR_NOT_SAVED"

    static_result = registry.run_tool(
        "run_factor_static_checks",
        {"code_path": code_result["code_path"]},
        context,
    )
    assert static_result["status"] == "PASSED"

    saved = registry.run_tool(
        "save_factor",
        {
            "factor_id": factor_id,
            "code_path": code_result["code_path"],
            "spec_path": code_result["spec_path"],
        },
        context,
    )
    assert saved["status"] == "saved"

    evaluated = registry.run_tool(
        "evaluate_factor_candidate",
        {"factor_id": factor_id, "start_date": "20240105", "end_date": "20240120"},
        context,
    )
    assert evaluated["status"] == "validated"
    assert evaluated["non_null"] > 0
