from __future__ import annotations

import threading
from types import SimpleNamespace
from typing import Any, ClassVar

from qmt_agent_trader.agent.cancellation import CancellationToken
from qmt_agent_trader.agent.llm_client import (
    Cancelled,
    DeepSeekClient,
    DeepSeekTool,
    FinalMessage,
    TextDelta,
    ToolCallComplete,
    ToolResult,
)


class _Delta:
    content = "partial"
    reasoning_content = None
    tool_calls: ClassVar[list[Any]] = []


class _Choice:
    def __init__(self) -> None:
        self.delta = _Delta()


class _Chunk:
    def __init__(self) -> None:
        self.choices = [_Choice()]


class _ClosableStream:
    def __init__(self, chunks: list[Any] | None = None) -> None:
        self.closed = False
        self.chunks = chunks or [_Chunk(), _Chunk()]

    def __iter__(self):
        return iter(self.chunks)

    def close(self) -> None:
        self.closed = True


class _StreamingCompletions:
    def __init__(self, stream: _ClosableStream) -> None:
        self.stream = stream
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> _ClosableStream:
        self.calls.append(kwargs)
        return self.stream


class _StreamingChat:
    def __init__(self, stream: _ClosableStream) -> None:
        self.completions = _StreamingCompletions(stream)


class _StreamingOpenAI:
    def __init__(self, stream: _ClosableStream) -> None:
        self.chat = _StreamingChat(stream)


class _BlockingClosableStream:
    def __init__(self) -> None:
        self.started = threading.Event()
        self.release = threading.Event()
        self.closed = False

    def __iter__(self):
        return self

    def __next__(self) -> Any:
        self.started.set()
        self.release.wait(timeout=2)
        if self.closed:
            raise StopIteration
        return _Chunk()

    def close(self) -> None:
        self.closed = True
        self.release.set()


def _client(stream: _ClosableStream) -> tuple[DeepSeekClient, _StreamingCompletions]:
    client = DeepSeekClient.__new__(DeepSeekClient)
    client.model = "deepseek-test"
    fake = _StreamingOpenAI(stream)
    client.client = fake
    return client, fake.chat.completions


def test_cancellation_before_model_request_does_not_call_model() -> None:
    stream = _ClosableStream()
    client, completions = _client(stream)
    token = CancellationToken()
    token.request_cancel()

    events = list(
        client.run_tool_loop_stream(
            messages=[{"role": "user", "content": "stop"}],
            tools=[],
            cancel_requested=token.is_cancel_requested,
        )
    )

    assert isinstance(events[-1], Cancelled)
    assert not completions.calls


def test_cancellation_inside_stream_closes_stream_without_final_message() -> None:
    stream = _ClosableStream()
    client, completions = _client(stream)
    token = CancellationToken()
    yielded = 0

    def cancel_requested() -> bool:
        nonlocal yielded
        yielded += 1
        if yielded >= 5:
            token.request_cancel()
        return token.is_cancel_requested()

    events = list(
        client.run_tool_loop_stream(
            messages=[{"role": "user", "content": "stop"}],
            tools=[],
            cancel_requested=cancel_requested,
        )
    )

    assert completions.calls
    assert stream.closed is True
    assert any(isinstance(event, TextDelta) for event in events)
    assert isinstance(events[-1], Cancelled)
    assert not any(event.__class__.__name__ == "FinalMessage" for event in events)


def _tool_chunk() -> Any:
    call = SimpleNamespace(
        index=0,
        id="call_lookup",
        function=SimpleNamespace(
            name="lookup",
            arguments='{"symbol": "000001.SZ"}',
        ),
    )
    delta = SimpleNamespace(content=None, reasoning_content=None, tool_calls=[call])
    return SimpleNamespace(choices=[SimpleNamespace(delta=delta)])


def test_cancellation_after_arguments_prevents_tool_execution() -> None:
    stream = _ClosableStream([_tool_chunk()])
    client, completions = _client(stream)
    token = CancellationToken()
    tool_calls = 0

    def lookup(**_kwargs: Any) -> dict[str, str]:
        nonlocal tool_calls
        tool_calls += 1
        return {"status": "ok"}

    events = []
    for event in client.run_tool_loop_stream(
        messages=[{"role": "user", "content": "stop before tool"}],
        tools=[
            DeepSeekTool(
                name="lookup",
                description="lookup",
                parameters={"type": "object", "properties": {}},
                fn=lookup,
            )
        ],
        cancel_requested=token.is_cancel_requested,
    ):
        events.append(event)
        if isinstance(event, ToolCallComplete):
            token.request_cancel()

    assert tool_calls == 0
    assert isinstance(events[-1], Cancelled)
    assert not any(isinstance(event, ToolResult) for event in events)
    assert not any(isinstance(event, FinalMessage) for event in events)
    assert completions.calls and stream.closed


def test_cancellation_after_sync_tool_returns_reports_tool_result_before_cancelled() -> None:
    stream = _ClosableStream([_tool_chunk()])
    client, completions = _client(stream)
    token = CancellationToken()

    def lookup(**_kwargs: Any) -> dict[str, str]:
        token.request_cancel()
        return {"status": "ok"}

    events = list(
        client.run_tool_loop_stream(
            messages=[{"role": "user", "content": "stop after tool"}],
            tools=[
                DeepSeekTool(
                    name="lookup",
                    description="lookup",
                    parameters={"type": "object", "properties": {}},
                    fn=lookup,
                )
            ],
            cancel_requested=token.is_cancel_requested,
        )
    )

    assert len(completions.calls) == 1
    assert isinstance(events[-1], Cancelled)
    assert any(isinstance(event, ToolResult) for event in events)
    assert not any(isinstance(event, FinalMessage) for event in events)
    assert stream.closed


def test_cancellation_after_tool_result_prevents_next_model_request() -> None:
    stream = _ClosableStream([_tool_chunk()])
    client, completions = _client(stream)
    token = CancellationToken()
    events = []

    for event in client.run_tool_loop_stream(
        messages=[{"role": "user", "content": "stop after result"}],
        tools=[
            DeepSeekTool(
                name="lookup",
                description="lookup",
                parameters={"type": "object", "properties": {}},
                fn=lambda **_kwargs: {"status": "ok"},
            )
        ],
        cancel_requested=token.is_cancel_requested,
    ):
        events.append(event)
        if isinstance(event, ToolResult):
            token.request_cancel()

    assert len(completions.calls) == 1
    assert any(isinstance(event, ToolResult) for event in events)
    assert isinstance(events[-1], Cancelled)
    assert not any(isinstance(event, FinalMessage) for event in events)


def test_cancellation_callback_closes_a_blocking_model_stream() -> None:
    stream = _BlockingClosableStream()
    client, completions = _client(stream)  # type: ignore[arg-type]
    token = CancellationToken()
    events: list[Any] = []

    worker = threading.Thread(
        target=lambda: events.extend(
            client.run_tool_loop_stream(
                messages=[{"role": "user", "content": "blocking"}],
                tools=[],
                cancel_requested=token,
            )
        )
    )
    worker.start()
    assert stream.started.wait(timeout=1)

    token.request_cancel()
    worker.join(timeout=1)

    assert not worker.is_alive()
    assert stream.closed is True
    assert completions.calls
    assert isinstance(events[-1], Cancelled)
    assert not any(isinstance(event, FinalMessage) for event in events)
