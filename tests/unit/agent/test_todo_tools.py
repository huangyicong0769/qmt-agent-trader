from __future__ import annotations

from qmt_agent_trader.agent.audit import AuditLogger
from qmt_agent_trader.agent.experiment_store import ExperimentStore
from qmt_agent_trader.agent.sandbox import CodeSandbox
from qmt_agent_trader.agent.schemas import ToolContext
from qmt_agent_trader.agent.tool_dependencies import AgentToolDependencies
from qmt_agent_trader.agent.tools.todo_tools import build_todo_tools
from qmt_agent_trader.core.config import Settings
from qmt_agent_trader.data.storage import DataLake


def _tools(tmp_path):
    settings = Settings(project_root=tmp_path)
    deps = AgentToolDependencies(
        settings=settings,
        data_lake=DataLake(
            root=settings.resolved_data_dir / "lake",
            duckdb_path=settings.resolved_data_dir / "db.duckdb",
        ),
        sandbox=CodeSandbox(),
        experiment_store=ExperimentStore(settings.resolved_data_dir / "experiments"),
        audit_logger=AuditLogger(settings.resolved_log_dir / "audit" / "tools.jsonl"),
    )
    return {tool.spec.name: tool for tool in build_todo_tools(deps)}


def test_todo_tools_share_state_within_session(tmp_path) -> None:
    tools = _tools(tmp_path)
    context = ToolContext(run_id="run_1", session_id="chat_1")

    created = tools["todo_set_list"].run(
        {"goal": "研究组合", "items": [{"title": "检查数据"}, {"title": "运行回测"}]},
        context,
    )
    first_id = created["todo_state"]["items"][0]["item_id"]
    tools["todo_update_item"].run(
        {"item_id": first_id, "status": "COMPLETED"},
        context,
    )
    status = tools["todo_get_status"].run({}, context)

    assert status["todo_state"]["session_id"] == "chat_1"
    assert status["todo_state"]["summary"]["total"] == 2
    assert status["todo_state"]["summary"]["completed"] == 1
    assert status["todo_state"]["schema_version"] == 2
    assert status["todo_state"]["revision"] >= 1


def test_todo_mutation_tool_schemas_expose_nonnegative_cas(tmp_path) -> None:
    tools = _tools(tmp_path)
    for name in (
        "todo_set_list", "todo_add_item", "todo_update_item", "todo_clear_completed"
    ):
        schema = tools[name].spec.input_schema["properties"]["expected_revision"]
        assert schema == {"type": "integer", "minimum": 0}


def test_todo_tools_isolate_different_sessions(tmp_path) -> None:
    tools = _tools(tmp_path)

    tools["todo_set_list"].run(
        {"items": [{"title": "A 会话任务"}]},
        ToolContext(run_id="run_1", session_id="chat_a"),
    )
    tools["todo_set_list"].run(
        {"items": [{"title": "B 会话任务"}]},
        ToolContext(run_id="run_2", session_id="chat_b"),
    )

    a_status = tools["todo_get_status"].run(
        {},
        ToolContext(run_id="run_3", session_id="chat_a"),
    )
    b_status = tools["todo_get_status"].run(
        {},
        ToolContext(run_id="run_4", session_id="chat_b"),
    )

    assert a_status["todo_state"]["items"][0]["title"] == "A 会话任务"
    assert b_status["todo_state"]["items"][0]["title"] == "B 会话任务"


def test_todo_clear_completed_keeps_pending_in_progress_and_blocked(tmp_path) -> None:
    tools = _tools(tmp_path)
    context = ToolContext(run_id="run_1", session_id="chat_1")
    created = tools["todo_set_list"].run(
        {
            "items": [
                {"title": "完成"},
                {"title": "进行中"},
                {"title": "阻塞"},
                {"title": "待处理"},
            ]
        },
        context,
    )
    ids = [item["item_id"] for item in created["todo_state"]["items"]]
    tools["todo_update_item"].run({"item_id": ids[0], "status": "COMPLETED"}, context)
    tools["todo_update_item"].run({"item_id": ids[1], "status": "IN_PROGRESS"}, context)
    tools["todo_update_item"].run({"item_id": ids[2], "status": "BLOCKED"}, context)

    cleared = tools["todo_clear_completed"].run({}, context)

    remaining_titles = [item["title"] for item in cleared["todo_state"]["items"]]
    assert remaining_titles == ["进行中", "阻塞", "待处理"]


def test_todo_completed_without_evidence_is_preserved_but_warns(tmp_path) -> None:
    tools = _tools(tmp_path)
    context = ToolContext(run_id="run_1", session_id="chat_1")
    created = tools["todo_set_list"].run({"items": [{"title": "运行回测"}]}, context)
    item_id = created["todo_state"]["items"][0]["item_id"]

    updated = tools["todo_update_item"].run(
        {"item_id": item_id, "status": "COMPLETED"},
        context,
    )

    assert updated["todo_state"]["items"][0]["status"] == "COMPLETED"
    assert updated["todo_consistency_status"] == "UNVERIFIED"
    assert "TODO_COMPLETION_UNVERIFIED" in updated["warnings"]


def test_todo_completed_with_failed_evidence_is_preserved_but_warns(tmp_path) -> None:
    tools = _tools(tmp_path)
    context = ToolContext(run_id="run_1", session_id="chat_1")
    created = tools["todo_set_list"].run({"items": [{"title": "运行回测"}]}, context)
    item_id = created["todo_state"]["items"][0]["item_id"]

    updated = tools["todo_update_item"].run(
        {
            "item_id": item_id,
            "status": "COMPLETED",
            "notes": "run_backtest diagnostics FAIL",
            "evidence_refs": ["run_backtest:research_1"],
        },
        context,
    )

    assert updated["todo_state"]["items"][0]["status"] == "COMPLETED"
    assert updated["todo_consistency_status"] == "CONFLICT"
    assert "TODO_COMPLETED_WITH_FAILED_EVIDENCE" in updated["warnings"]
