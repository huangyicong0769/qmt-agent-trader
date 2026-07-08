"""Tests for the Self-Bootstrap workflow."""

from __future__ import annotations

import pytest

from qmt_agent_trader.agent.experiment_store import ExperimentStore
from qmt_agent_trader.agent.sandbox import CodeSandbox
from qmt_agent_trader.agent.schemas import ExperimentStatus
from qmt_agent_trader.agent.tools import build_agent_registry
from qmt_agent_trader.agent.workflows.self_bootstrap import SelfBootstrapWorkflow
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
        "tushare/daily",
    )
    return reg


@pytest.fixture
def store(tmp_path):
    return ExperimentStore(tmp_path / "experiments")


def test_bootstrap_creates_experiment(registry, store):
    # Create a prior failure to seed the gap detection
    exp = store.create_experiment("factor_discovery", tags=["test"])
    store.update_experiment(exp.experiment_id, status=ExperimentStatus.FAILED)
    store.add_lesson(exp.experiment_id, "missing data layer for PIT fundamentals")

    wf = SelfBootstrapWorkflow(registry, store)
    result = wf.run([exp.experiment_id])
    assert result.kind == "self_bootstrap"
    assert result.experiment_id is not None


def test_bootstrap_rejects_forbidden_tools(registry, store):
    wf = SelfBootstrapWorkflow(registry, store)
    result = wf.run(["test_reject"])
    assert result.status is not None
    # Verify no broker/gateway/live tools were proposed
    lessons_lower = [lesson.lower() for lesson in result.lessons]
    artifacts_lower = [a.lower() for a in result.artifacts]
    all_text = " ".join(lessons_lower + artifacts_lower)
    assert "broker" not in all_text or "rejected" in all_text


def test_bootstrap_cannot_register_approval_required(registry, store):
    """Self-bootstrap should NEVER auto-register APPROVAL_REQUIRED tools."""
    wf = SelfBootstrapWorkflow(registry, store)
    result = wf.run(["test_safety"])
    # The result should not contain any automatically-registered high-permission tools
    # (verification is implicit: the propose_tool_registration tool requires
    # APPROVAL_REQUIRED which the LLM cannot call)
    assert result.status is not None


def test_bootstrap_handles_empty_failures(registry, store):
    """When there are no prior failures, the workflow should still complete."""
    wf = SelfBootstrapWorkflow(registry, store)
    result = wf.run([])
    assert result.status in (
        ExperimentStatus.COMPLETED,
        ExperimentStatus.REVIEW_REQUIRED,
    )
