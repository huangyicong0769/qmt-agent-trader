from __future__ import annotations

import threading
import time
from typing import ClassVar

import anyio
from pydantic import SecretStr

from qmt_agent_trader.agent.cancellation import CancellationToken
from qmt_agent_trader.agent.llm_client import (
    Cancelled,
    FinalMessage,
    TextDelta,
    ToolCallComplete,
    ToolResult,
)
from qmt_agent_trader.agent.orchestrator import AgentOrchestrator
from qmt_agent_trader.core.config import Settings
from qmt_agent_trader.data.storage import DataLake


class _CancellableClient:
    seen_cancel_callbacks: ClassVar[list[object]] = []

    def __init__(self, **_kwargs: object) -> None:
        pass

    def run_tool_loop_stream(self, *, messages, tools, max_rounds, cancel_requested=None):
        self.seen_cancel_callbacks.append(cancel_requested)
        yield TextDelta(content="worker event")
        while cancel_requested is not None and not cancel_requested():
            time.sleep(0.001)
        yield Cancelled()


class _NormalClient:
    def __init__(self, **_kwargs: object) -> None:
        pass

    def run_tool_loop_stream(self, *, messages, tools, max_rounds, cancel_requested=None):
        yield FinalMessage(content="完成")


class _ErrorClient:
    def __init__(self, **_kwargs: object) -> None:
        pass

    def run_tool_loop_stream(self, *, messages, tools, max_rounds, cancel_requested=None):
        raise RuntimeError("worker exploded")


class _ToolThenWaitClient:
    def __init__(self, **_kwargs: object) -> None:
        pass

    def run_tool_loop_stream(self, *, messages, tools, max_rounds, cancel_requested=None):
        yield ToolCallComplete(
            tool_call_id="call-1",
            tool_name="lookup",
            arguments={"symbol": "000001.SZ"},
        )
        yield ToolResult(
            tool_call_id="call-1",
            tool_name="lookup",
            result={"status": "ok", "value": 1},
        )
        while cancel_requested is None or not cancel_requested():
            time.sleep(0.001)
        yield Cancelled()


def _orchestrator(tmp_path) -> AgentOrchestrator:
    settings = Settings(project_root=tmp_path, deepseek_api_key=SecretStr("key"))
    lake = DataLake(root=tmp_path / "lake", duckdb_path=tmp_path / "db.duckdb")
    return AgentOrchestrator(settings=settings, data_lake=lake)


def test_worker_events_cross_thread_queue_and_cooperative_cancel(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(
        "qmt_agent_trader.agent.orchestrator.DeepSeekClient",
        _CancellableClient,
    )
    orchestrator = _orchestrator(tmp_path)
    nonlocal_token = [False]

    async def collect_with_token() -> list[object]:
        events: list[object] = []

        def cancel_requested() -> bool:
            return nonlocal_token[0]

        async for event in orchestrator.execute_stream(
            "长任务",
            run_id="run-worker-cancel",
            cancel_requested=cancel_requested,
        ):
            events.append(event)
            if event.type == "token":
                nonlocal_token[0] = True
        return events

    events = anyio.run(collect_with_token)
    assert any(event.type == "cancelled" for event in events)
    assert all(event.type != "done" for event in events)


def test_worker_exception_becomes_one_error_event(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(
        "qmt_agent_trader.agent.orchestrator.DeepSeekClient",
        _ErrorClient,
    )
    events = anyio.run(
        lambda: _collect(
            _orchestrator(tmp_path),
            run_id="run-worker-error",
        )
    )

    errors = [event for event in events if event.type == "error"]
    assert len(errors) == 1
    assert all(event.type != "done" for event in events)


def test_worker_normal_completion_has_one_terminal_done(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(
        "qmt_agent_trader.agent.orchestrator.DeepSeekClient",
        _NormalClient,
    )
    events = anyio.run(
        lambda: _collect(
            _orchestrator(tmp_path),
            run_id="run-worker-done",
        )
    )

    assert [event.type for event in events[-2:]] == ["final_message", "done"]
    assert sum(event.type == "done" for event in events) == 1


def test_cancel_wins_over_worker_error_fallback(monkeypatch, tmp_path) -> None:
    cancelled = [False]

    class _ErrorAfterToolClient:
        def __init__(self, **_kwargs: object) -> None:
            pass

        def run_tool_loop_stream(
            self,
            *,
            messages,
            tools,
            max_rounds,
            cancel_requested=None,
        ):
            yield ToolResult(
                tool_call_id="call-1",
                tool_name="lookup",
                result={"status": "ok"},
            )
            cancelled[0] = True
            raise RuntimeError("stream failed after cancellation")

    monkeypatch.setattr(
        "qmt_agent_trader.agent.orchestrator.DeepSeekClient",
        _ErrorAfterToolClient,
    )
    events = anyio.run(
        lambda: _collect(
            _orchestrator(tmp_path),
            run_id="run-cancel-error-fallback",
            cancel_requested=lambda: cancelled[0],
        )
    )

    assert any(event.type == "cancelled" for event in events)
    assert not any(event.type == "final_message" for event in events)
    assert not any(event.type == "done" for event in events)


def test_completed_tool_result_reaches_orchestrator_before_cancel(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(
        "qmt_agent_trader.agent.orchestrator.DeepSeekClient",
        _ToolThenWaitClient,
    )
    cancellation_requested = [False]

    async def collect() -> list[object]:
        events: list[object] = []
        async for event in _orchestrator(tmp_path).execute_stream(
            "工具后停止",
            run_id="run-tool-cancel",
            cancel_requested=lambda: cancellation_requested[0],
        ):
            events.append(event)
            if event.type == "tool_done":
                cancellation_requested[0] = True
        return events

    events = anyio.run(collect)
    event_types = [event.type for event in events]
    assert event_types.index("tool_args") < event_types.index("tool_done") < event_types.index(
        "cancelled"
    )
    assert "final_message" not in event_types
    assert "done" not in event_types


def test_closing_orchestrator_stream_waits_for_worker_exit(monkeypatch, tmp_path) -> None:
    worker_finished = threading.Event()

    class _BlockingWorkerClient:
        def __init__(self, **_kwargs: object) -> None:
            pass

        def run_tool_loop_stream(
            self,
            *,
            messages,
            tools,
            max_rounds,
            cancel_requested=None,
        ):
            yield TextDelta(content="worker started")
            while cancel_requested is not None and not cancel_requested():
                time.sleep(0.001)
            worker_finished.set()
            yield Cancelled()

    monkeypatch.setattr(
        "qmt_agent_trader.agent.orchestrator.DeepSeekClient",
        _BlockingWorkerClient,
    )
    token = CancellationToken()

    async def close_stream() -> None:
        orchestrator = _orchestrator(tmp_path)
        stream = orchestrator.execute_stream(
            "关闭订阅",
            run_id="run-close-worker",
            cancel_requested=token,
        )
        assert (await anext(stream)).type == "run_started"
        assert (await anext(stream)).type == "progress"
        assert (await anext(stream)).type == "token"
        token.request_cancel()
        await stream.aclose()

    anyio.run(close_stream)
    assert worker_finished.is_set()


async def _collect(
    orchestrator: AgentOrchestrator,
    *,
    run_id: str,
    cancel_requested=None,
) -> list[object]:
    return [
        event
        async for event in orchestrator.execute_stream(
            "测试",
            run_id=run_id,
            cancel_requested=cancel_requested,
        )
    ]
