"""Tests for agent.experiment_store."""

from __future__ import annotations

import pytest

from qmt_agent_trader.agent.errors import ExperimentNotFoundError
from qmt_agent_trader.agent.experiment_store import ExperimentStore
from qmt_agent_trader.agent.sandbox import CodeSandbox
from qmt_agent_trader.agent.schemas import ExperimentStatus, ToolContext
from qmt_agent_trader.agent.tools import build_agent_registry
from qmt_agent_trader.agent.tools.experiment_tools import (
    log_experiment_event_tool,
    search_experiments_tool,
    set_experiment_store,
)
from qmt_agent_trader.data.storage import DataLake


@pytest.fixture
def store(tmp_path):
    return ExperimentStore(tmp_path / "experiments")


def test_create_experiment(store: ExperimentStore) -> None:
    exp = store.create_experiment("factor_discovery", tags=["mom"])
    assert exp.kind == "factor_discovery"
    assert exp.status == ExperimentStatus.CREATED
    assert exp.tags == ["mom"]


def test_create_experiment_with_explicit_id(store: ExperimentStore) -> None:
    exp = store.create_experiment("factor_discovery", experiment_id="exp_context")

    assert exp.experiment_id == "exp_context"
    assert store.get_experiment("exp_context").kind == "factor_discovery"


def test_get_experiment(store: ExperimentStore) -> None:
    exp = store.create_experiment("strategy_engineering")
    fetched = store.get_experiment(exp.experiment_id)
    assert fetched.experiment_id == exp.experiment_id


def test_get_missing_raises(store: ExperimentStore) -> None:
    with pytest.raises(ExperimentNotFoundError):
        store.get_experiment("exp_nope")


def test_update_status(store: ExperimentStore) -> None:
    exp = store.create_experiment("factor_discovery")
    updated = store.update_experiment(
        exp.experiment_id, status=ExperimentStatus.RUNNING
    )
    assert updated.status == ExperimentStatus.RUNNING


def test_add_lesson(store: ExperimentStore) -> None:
    exp = store.create_experiment("factor_discovery")
    store.add_lesson(exp.experiment_id, "failed due to NaN coverage")
    fetched = store.get_experiment(exp.experiment_id)
    assert "failed due to NaN coverage" in fetched.lessons


def test_search_tool_reports_degraded_corrupt_records(store: ExperimentStore) -> None:
    set_experiment_store(store)
    store.root.mkdir(parents=True, exist_ok=True)
    (store.root / "exp_broken.json").write_text("{broken", encoding="utf-8")
    result = search_experiments_tool.run({}, ToolContext(run_id="run_diag"))
    assert result["status"] == "DEGRADED"
    assert len(result["diagnostics"]) == 1


def test_log_experiment_event_uses_context_id(store: ExperimentStore) -> None:
    set_experiment_store(store)
    store.create_experiment("factor_discovery", experiment_id="exp_context")

    result = log_experiment_event_tool.run(
        {"event_type": "observation", "message": "context works"},
        ToolContext(run_id="run_context", experiment_id="exp_context"),
    )

    assert result["status"] == "logged"
    assert "[observation] context works" in store.get_experiment("exp_context").lessons


def test_registry_preserves_explicit_experiment_root(tmp_path) -> None:
    explicit = tmp_path / "explicit-experiments"
    ExperimentStore(explicit).create_experiment(
        "factor", experiment_id="exp_explicit", tags=["explicit"]
    )
    registry = build_agent_registry(
        data_lake=DataLake(root=tmp_path / "lake", duckdb_path=tmp_path / "db.duckdb"),
        audit_path=tmp_path / "audit.jsonl",
        experiment_root=explicit,
        sandbox=CodeSandbox(tmp_path / "generated"),
    )
    result = registry.run_tool(
        "search_experiments", {"tags": ["explicit"]}, ToolContext(run_id="run_explicit")
    )
    assert [item["experiment_id"] for item in result["experiments"]] == ["exp_explicit"]


def test_get_experiment_tool_calls_returns_real_audit_entries(tmp_path) -> None:
    lake = DataLake(
        root=tmp_path / "lake",
        duckdb_path=tmp_path / "test.duckdb",
    )
    registry = build_agent_registry(
        data_lake=lake,
        audit_path=tmp_path / "audit.jsonl",
        experiment_root=tmp_path / "experiments",
        sandbox=CodeSandbox(tmp_path / "generated"),
    )
    context = ToolContext(
        run_id="run_audit",
        session_id="session_audit",
        experiment_id="exp_audit",
    )
    other_context = ToolContext(
        run_id="run_other",
        session_id="session_other",
        experiment_id="exp_other",
    )

    registry.run_tool("list_strategy_candidates", {"query": "none"}, context)
    registry.run_tool("list_strategy_candidates", {"query": "none"}, other_context)
    result = registry.run_tool("get_experiment_tool_calls", {}, context)
    current_alias = registry.run_tool(
        "get_experiment_tool_calls",
        {"session_id": "current"},
        context,
    )

    assert result["status"] == "ok"
    assert result["session_id"] == "session_audit"
    assert result["count"] == 1
    assert result["tool_calls"][0]["tool_name"] == "list_strategy_candidates"
    assert result["tool_calls"][0]["session_id"] == "session_audit"
    assert result["tool_calls"][0]["output"]["status"] == "ok"
    assert current_alias["session_id"] == "session_audit"
    assert current_alias["count"] == 2


def test_get_experiment_tool_calls_requests_range_when_result_is_too_large(tmp_path) -> None:
    lake = DataLake(
        root=tmp_path / "lake",
        duckdb_path=tmp_path / "test.duckdb",
    )
    registry = build_agent_registry(
        data_lake=lake,
        audit_path=tmp_path / "audit.jsonl",
        experiment_root=tmp_path / "experiments",
        sandbox=CodeSandbox(tmp_path / "generated"),
    )
    context = ToolContext(
        run_id="run_audit",
        session_id="session_audit",
        experiment_id="exp_audit",
    )
    for index in range(45):
        registry.run_tool(
            "list_strategy_candidates",
            {"query": f"strategy-{index:02d}"},
            context,
        )

    result = registry.run_tool("get_experiment_tool_calls", {}, context)

    assert result["status"] == "RANGE_REQUIRED"
    assert result["total_count"] == 45
    assert result["record_numbering"] == "1-based oldest-to-newest"
    assert result["range_parameter"] == "record_range"
    assert result["suggested_ranges"] == [
        {"record_range": {"start": 1, "end": 15}},
        {"record_range": {"start": 16, "end": 30}},
        {"record_range": {"start": 31, "end": 45}},
    ]
    assert "tool_calls" not in result


def test_get_experiment_tool_calls_returns_requested_record_range(tmp_path) -> None:
    lake = DataLake(
        root=tmp_path / "lake",
        duckdb_path=tmp_path / "test.duckdb",
    )
    registry = build_agent_registry(
        data_lake=lake,
        audit_path=tmp_path / "audit.jsonl",
        experiment_root=tmp_path / "experiments",
        sandbox=CodeSandbox(tmp_path / "generated"),
    )
    context = ToolContext(
        run_id="run_audit",
        session_id="session_audit",
        experiment_id="exp_audit",
    )
    for index in range(45):
        registry.run_tool(
            "list_strategy_candidates",
            {"query": f"strategy-{index:02d}"},
            context,
        )

    result = registry.run_tool(
        "get_experiment_tool_calls",
        {"record_range": {"start": 16, "end": 30}},
        context,
    )

    assert result["status"] == "ok"
    assert result["total_count"] == 45
    assert result["record_range"] == {"start": 16, "end": 30}
    assert result["returned_count"] == 15
    assert result["has_previous_range"] is True
    assert result["has_next_range"] is True
    assert result["tool_calls"][0]["record_number"] == 16
    assert result["tool_calls"][-1]["record_number"] == 30
    assert [item["tool_name"] for item in result["tool_calls"]] == [
        "list_strategy_candidates"
    ] * 15


def test_get_experiment_tool_calls_schema_exposes_record_range(tmp_path) -> None:
    schema = build_agent_registry(
        data_lake=DataLake(
            root=tmp_path / "lake",
            duckdb_path=tmp_path / "test.duckdb",
        ),
        audit_path=tmp_path / "audit.jsonl",
        experiment_root=tmp_path / "experiments",
        sandbox=CodeSandbox(tmp_path / "generated"),
    ).describe_tool("get_experiment_tool_calls").input_schema

    assert "record_range" in schema["properties"]
    assert "limit" not in schema["properties"]


def test_add_artifact(store: ExperimentStore) -> None:
    exp = store.create_experiment("strategy_engineering")
    store.add_artifact(exp.experiment_id, "/path/to/code.py")
    fetched = store.get_experiment(exp.experiment_id)
    assert "/path/to/code.py" in fetched.artifacts


def test_search_by_tag(store: ExperimentStore) -> None:
    store.create_experiment("a", tags=["mom"])
    store.create_experiment("b", tags=["vol"])
    results = store.search_experiments(tags=["mom"])
    assert len(results) == 1
    assert results[0].tags == ["mom"]


def test_search_by_query(store: ExperimentStore) -> None:
    store.create_experiment("a", hypothesis={"desc": "momentum test"})
    store.create_experiment("b", hypothesis={"desc": "volatility test"})
    results = store.search_experiments(query="momentum")
    assert len(results) == 1


def test_list_recent_failures(store: ExperimentStore) -> None:
    exp = store.create_experiment("x")
    store.update_experiment(exp.experiment_id, status=ExperimentStatus.FAILED)
    failures = store.list_recent_failures()
    assert len(failures) >= 1
    assert failures[0].status == ExperimentStatus.FAILED
