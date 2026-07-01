from __future__ import annotations

from typing import ClassVar

import anyio
from pydantic import SecretStr

from qmt_agent_trader.agent.llm_client import FinalMessage, SafetyCapHit, TextDelta, ToolResult
from qmt_agent_trader.agent.orchestrator import (
    AgentOrchestrator,
    _is_data_acquisition_request,
    _is_large_batch_data_request,
    _requires_fresh_evidence,
)
from qmt_agent_trader.core.config import Settings
from qmt_agent_trader.data.storage import DataLake


class CapturingDeepSeekClient:
    seen_messages: ClassVar[list[dict]] = []
    seen_tool_names: ClassVar[list[str]] = []

    def __init__(self, **_kwargs: object) -> None:
        pass

    def run_tool_loop_stream(self, *, messages: list[dict], tools: list[object], max_rounds: int):
        CapturingDeepSeekClient.seen_messages = messages
        CapturingDeepSeekClient.seen_tool_names = [tool.name for tool in tools]
        return iter(())


def test_execute_stream_includes_recent_natural_session_history(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(
        "qmt_agent_trader.agent.orchestrator.DeepSeekClient",
        CapturingDeepSeekClient,
    )
    settings = Settings(project_root=tmp_path, deepseek_api_key=SecretStr("key"))
    lake = DataLake(root=tmp_path / "lake", duckdb_path=tmp_path / "db.duckdb")
    orchestrator = AgentOrchestrator(settings=settings, data_lake=lake)

    async def collect_events() -> list[object]:
        return [
            event
            async for event in orchestrator.execute_stream(
                "继续回答",
                history=[
                    {"role": "user", "content": "先看 159259"},
                    {"role": "tool", "content": "ignored"},
                    {"role": "assistant", "content": "159259 是 ETF"},
                    {"role": "done", "content": "ignored"},
                    {"role": "user", "content": "继续回答"},
                ],
                run_id="run-test",
            )
        ]

    events = anyio.run(collect_events)

    assert events[-1].type == "done"
    assert CapturingDeepSeekClient.seen_messages[-3:] == [
        {"role": "user", "content": "先看 159259"},
        {"role": "assistant", "content": "159259 是 ETF"},
        {"role": "user", "content": "继续回答"},
    ]
    system_prompt = CapturingDeepSeekClient.seen_messages[0]["content"]
    assert "actual_data_end" in system_prompt
    assert "stale_vs_requested_end" in system_prompt
    assert "dry_run" in system_prompt
    assert "retry, retest, rerun, verify again" in system_prompt
    assert "research-only" in system_prompt
    assert "coverage_end_date" in system_prompt
    assert "list_saved_factors" in system_prompt
    assert "requires_trade_calendar_validation" in system_prompt
    assert "CALENDAR_VALIDATION_REQUIRED" in system_prompt
    assert "PARTIAL_COVERAGE" in system_prompt
    assert "missing_symbols" in system_prompt
    assert "stale_symbols" in system_prompt
    assert "For data acquisition or coverage-check requests" in system_prompt
    assert "do not stop after a dry_run plan or ask whether to fetch" in system_prompt
    assert "For large-basket or bulk data pulls" in system_prompt
    assert "pass the full symbols=[...] list" in system_prompt
    assert "obey the update tool's live-execution scope" in system_prompt
    assert "If a live basket fill is not supported" in system_prompt
    assert "Do not blame replay, validation, or test protocols" in system_prompt
    assert "Do not expand the user-requested date window" in system_prompt
    assert "Do not validate only a sample symbol" in system_prompt
    assert "intent has been classified" not in system_prompt
    assert "Suggested workflow" not in system_prompt
    assert "Recommended tools" not in system_prompt


class SafetyCapDeepSeekClient:
    def __init__(self, **_kwargs: object) -> None:
        pass

    def run_tool_loop_stream(self, *, messages: list[dict], tools: list[object], max_rounds: int):
        yield TextDelta(content="正在评估")
        yield SafetyCapHit(message="Safety cap: LLM tool loop reached 24 rounds without finishing.")


def test_execute_stream_reports_safety_cap_as_error_without_done(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(
        "qmt_agent_trader.agent.orchestrator.DeepSeekClient",
        SafetyCapDeepSeekClient,
    )
    settings = Settings(project_root=tmp_path, deepseek_api_key=SecretStr("key"))
    lake = DataLake(root=tmp_path / "lake", duckdb_path=tmp_path / "db.duckdb")
    orchestrator = AgentOrchestrator(settings=settings, data_lake=lake)

    async def collect_events() -> list[object]:
        return [
            event
            async for event in orchestrator.execute_stream(
                "在这个ETF上寻找有效因子，并进行评估",
                run_id="run-safety-cap",
                max_rounds=24,
            )
        ]

    events = anyio.run(collect_events)

    assert any(event.type == "error" for event in events)
    assert all(event.type != "done" for event in events)


class TextOnlyDeepSeekClient:
    seen_messages: ClassVar[list[dict]] = []

    def __init__(self, **_kwargs: object) -> None:
        pass

    def run_tool_loop_stream(self, *, messages: list[dict], tools: list[object], max_rounds: int):
        TextOnlyDeepSeekClient.seen_messages = messages
        yield TextDelta(content="我重新检查了，现在没有问题。")
        yield FinalMessage(content="我重新检查了，现在没有问题。")


def test_execute_stream_guides_retry_requests_without_runtime_rejection(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setattr(
        "qmt_agent_trader.agent.orchestrator.DeepSeekClient",
        TextOnlyDeepSeekClient,
    )
    settings = Settings(project_root=tmp_path, deepseek_api_key=SecretStr("key"))
    lake = DataLake(root=tmp_path / "lake", duckdb_path=tmp_path / "db.duckdb")
    orchestrator = AgentOrchestrator(settings=settings, data_lake=lake)

    async def collect_events() -> list[object]:
        return [
            event
            async for event in orchestrator.execute_stream(
                "tool 可能出错了，修复后再试试",
                run_id="run-fresh-evidence",
            )
        ]

    events = anyio.run(collect_events)

    assert [event.type for event in events[-2:]] == ["final_message", "done"]
    system_prompt = TextOnlyDeepSeekClient.seen_messages[0]["content"]
    assert "Fresh evidence is likely needed for this request" in system_prompt
    assert "Prefer existing conversation/tool evidence" in system_prompt


class QueryOnlyDecisionDeepSeekClient:
    seen_messages: ClassVar[list[dict]] = []

    def __init__(self, **_kwargs: object) -> None:
        pass

    def run_tool_loop_stream(self, *, messages: list[dict], tools: list[object], max_rounds: int):
        QueryOnlyDecisionDeepSeekClient.seen_messages = messages
        yield ToolResult(
            tool_call_id="call-1",
            tool_name="query_bars",
            result={"rows": [{"symbol": "159259.SZ", "trade_date": "20260626"}]},
        )
        yield TextDelta(content="建议买入。")
        yield FinalMessage(content="建议买入。")


class TodoToolDeepSeekClient:
    def __init__(self, **_kwargs: object) -> None:
        pass

    def run_tool_loop_stream(self, *, messages: list[dict], tools: list[object], max_rounds: int):
        todo_tool = next(tool for tool in tools if tool.name == "todo_set_list")
        result = todo_tool.fn(items=[{"title": "检查数据"}, {"title": "运行回测"}])
        yield ToolResult(
            tool_call_id="call-todo",
            tool_name="todo_set_list",
            result=result,
        )
        yield FinalMessage(content="我会按清单执行。")


def test_execute_stream_emits_todo_status_with_session_id(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(
        "qmt_agent_trader.agent.orchestrator.DeepSeekClient",
        TodoToolDeepSeekClient,
    )
    settings = Settings(project_root=tmp_path, deepseek_api_key=SecretStr("key"))
    lake = DataLake(root=tmp_path / "lake", duckdb_path=tmp_path / "db.duckdb")
    orchestrator = AgentOrchestrator(settings=settings, data_lake=lake)

    async def collect_events() -> list[object]:
        return [
            event
            async for event in orchestrator.execute_stream(
                "制定一个研究计划",
                run_id="run-todo",
                session_id="chat_x",
            )
        ]

    events = anyio.run(collect_events)
    event_types = [event.type for event in events]
    todo_events = [event for event in events if event.type == "todo_status"]

    assert "tool_done" in event_types
    assert len(todo_events) == 1
    assert todo_events[0].data["session_id"] == "chat_x"
    assert todo_events[0].data["summary"]["total"] == 2
    assert todo_events[0].data["items"][0]["title"] == "检查数据"


def test_execute_stream_guides_trade_decisions_without_runtime_rejection(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setattr(
        "qmt_agent_trader.agent.orchestrator.DeepSeekClient",
        QueryOnlyDecisionDeepSeekClient,
    )
    settings = Settings(project_root=tmp_path, deepseek_api_key=SecretStr("key"))
    lake = DataLake(root=tmp_path / "lake", duckdb_path=tmp_path / "db.duckdb")
    orchestrator = AgentOrchestrator(settings=settings, data_lake=lake)

    async def collect_events() -> list[object]:
        return [
            event
            async for event in orchestrator.execute_stream(
                "基于这些因子和当前行情，下个交易日应该买还是卖？",
                run_id="run-trade-evidence",
            )
        ]

    events = anyio.run(collect_events)

    assert [event.type for event in events[-2:]] == ["final_message", "done"]
    system_prompt = QueryOnlyDecisionDeepSeekClient.seen_messages[0]["content"]
    assert "This is a trade/risk decision request" in system_prompt
    assert "list_saved_factors" in system_prompt


def test_requires_fresh_evidence_detects_session5_retry_and_trade_prompts() -> None:
    assert _requires_fresh_evidence("可能是tool出错了，检查并修复后再试试")
    assert _requires_fresh_evidence("基于这些因子和当前行情，下个交易日应该买还是卖")
    assert not _requires_fresh_evidence("继续回答")


def test_data_acquisition_detection_covers_combo_coverage_prompts() -> None:
    assert _is_data_acquisition_request(
        "检查一下你现在是否能正常获取这个标的组合自上市以来的数据"
    )
    assert _is_data_acquisition_request("补齐 600519.SH 和 000858.SZ 的远程数据")
    assert not _is_data_acquisition_request("在这个组合上寻找有效因子")


def test_large_batch_detection_covers_bulk_symbol_fetch_prompts() -> None:
    assert _is_large_batch_data_request("测试大批量标的数据拉取，54个工业金属标的")
    assert _is_large_batch_data_request("批量拉取几十个标的的日线数据")
    assert _is_large_batch_data_request("large basket bulk data pull for 50 symbols")
    assert not _is_large_batch_data_request("补齐 600519.SH 的远程数据")


def test_execute_stream_guides_large_batch_data_pull_without_hiding_agent_tools(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setattr(
        "qmt_agent_trader.agent.orchestrator.DeepSeekClient",
        CapturingDeepSeekClient,
    )
    settings = Settings(project_root=tmp_path, deepseek_api_key=SecretStr("key"))
    lake = DataLake(root=tmp_path / "lake", duckdb_path=tmp_path / "db.duckdb")
    orchestrator = AgentOrchestrator(settings=settings, data_lake=lake)

    async def collect_events() -> list[object]:
        return [
            event
            async for event in orchestrator.execute_stream(
                "测试大批量标的数据拉取，54个工业金属标的，批量拉取日线并验证",
                run_id="run-bulk-data-tools",
            )
        ]

    events = anyio.run(collect_events)

    assert events[-1].type == "done"
    system_prompt = CapturingDeepSeekClient.seen_messages[0]["content"]
    assert "For large-basket or bulk data pulls" in system_prompt
    assert "obey the update tool's live-execution scope" in system_prompt
    assert "run_remote_data_update" in CapturingDeepSeekClient.seen_tool_names
    assert "query_bars" in CapturingDeepSeekClient.seen_tool_names
    assert "run_backtest" in CapturingDeepSeekClient.seen_tool_names
    assert "detect_tool_gap" in CapturingDeepSeekClient.seen_tool_names
