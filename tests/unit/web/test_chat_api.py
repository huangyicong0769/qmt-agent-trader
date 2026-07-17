"""Tests for chat API routes — natural language, no forced mode."""

from __future__ import annotations

import asyncio
import json

import anyio
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.requests import Request

from qmt_agent_trader.agent.orchestrator import OrchestratorEvent
from qmt_agent_trader.web.chat_repository import ChatSessionRepository
from qmt_agent_trader.web.chat_run_manager import ChatRunManager
from qmt_agent_trader.web.event_bus import EventBus
from qmt_agent_trader.web.routes import chat


@pytest.fixture(autouse=True)
def isolated_chat_repository(tmp_path, monkeypatch) -> ChatSessionRepository:
    repository = ChatSessionRepository(tmp_path / "sessions")
    monkeypatch.setattr(chat, "get_chat_session_repository", lambda: repository)
    return repository


def test_chat_api_create_and_send_message() -> None:
    app = FastAPI()
    app.include_router(chat.router)
    client = TestClient(app)

    # No mode required
    created = client.post("/sessions", json={"title": "Research"}).json()
    session_id = created["session_id"]

    # Send natural language message — no mode param
    response = client.post(
        f"/sessions/{session_id}/messages",
        json={"content": "帮我发现低波动因子"},
    )
    assert response.status_code == 200
    payload = response.json()

    assert "routing_decision" not in payload
    assert "message" in payload
    assert "session_id" in payload
    assert "run_id" in payload
    assert "llm_configured" in payload
    assert payload["message"]["role"] == "user"
    assert payload["message"]["content"] == "帮我发现低波动因子"


def test_chat_api_lists_sessions() -> None:
    app = FastAPI()
    app.include_router(chat.router)
    client = TestClient(app)
    client.post("/sessions", json={"title": "One"})

    response = client.get("/sessions")

    assert response.status_code == 200
    assert len(response.json()) == 1


def test_send_message_without_mode_works() -> None:
    """Sending a message without mode should not fail."""
    app = FastAPI()
    app.include_router(chat.router)
    client = TestClient(app)

    session = client.post("/sessions", json={"title": "T"}).json()
    resp = client.post(
        f"/sessions/{session['session_id']}/messages",
        json={"content": "解释回测结果"},
    )
    assert resp.status_code == 200
    payload = resp.json()
    assert "routing_decision" not in payload
    assert payload["message"]["role"] == "user"
    assert payload["message"]["content"] == "解释回测结果"


def test_send_message_with_advanced_options() -> None:
    """Advanced options should be accepted but not required."""
    app = FastAPI()
    app.include_router(chat.router)
    client = TestClient(app)

    session = client.post("/sessions", json={}).json()
    resp = client.post(
        f"/sessions/{session['session_id']}/messages",
        json={
            "content": "发现因子",
            "advanced": {"universe": "stock", "budget_mode": "fast"},
        },
    )
    assert resp.status_code == 200


def test_execute_without_body_reuses_queued_message_without_duplicate(
    isolated_chat_repository: ChatSessionRepository,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Executing an enqueued message should not append the same user text again."""
    class FakeOrchestrator:
        async def execute_stream(self, message: str, **kwargs: object):
            yield OrchestratorEvent(
                type="done",
                run_id=str(kwargs["run_id"]),
                session_id=str(kwargs["session_id"]),
                message="完成",
            )

    manager = ChatRunManager(
        orchestrator=FakeOrchestrator(),
        repository=isolated_chat_repository,
    )
    monkeypatch.setattr(chat, "_get_run_manager", lambda: manager)
    app = FastAPI()
    app.include_router(chat.router)
    client = TestClient(app)

    session = client.post("/sessions", json={}).json()
    session_id = session["session_id"]
    client.post(
        f"/sessions/{session_id}/messages",
        json={"content": "发现因子"},
    )

    async def call_execute() -> None:
        async def receive() -> dict[str, object]:
            return {"type": "http.request", "body": b"{}", "more_body": False}

        request = Request(
            {
                "type": "http",
                "method": "POST",
                "path": f"/sessions/{session_id}/execute",
                "headers": [(b"content-type", b"application/json")],
            },
            receive=receive,
        )
        await chat.execute_stream(session_id, request)

    anyio.run(call_execute)
    payload = client.get(f"/sessions/{session_id}").json()
    user_messages = [message for message in payload["messages"] if message["role"] == "user"]
    assert [message["content"] for message in user_messages] == ["发现因子"]


def test_execute_stream_outputs_todo_status_event_with_session_id(
    isolated_chat_repository: ChatSessionRepository,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = FastAPI()
    app.include_router(chat.router)
    client = TestClient(app)
    session = client.post("/sessions", json={}).json()
    session_id = session["session_id"]

    class FakeOrchestrator:
        async def execute_stream(self, message: str, **kwargs: object):
            assert message == "制定计划"
            assert kwargs["session_id"] == session_id
            yield OrchestratorEvent(
                type="todo_status",
                run_id="run_test",
                message="Todo status: 0/1 completed",
                data={
                    "session_id": session_id,
                    "items": [{"title": "检查数据", "status": "PENDING"}],
                    "summary": {"total": 1, "completed": 0},
                    "active_item": None,
                },
            )
            yield OrchestratorEvent(
                type="done",
                run_id="run_test",
                session_id=session_id,
                message="完成",
            )

    manager = ChatRunManager(
        orchestrator=FakeOrchestrator(),
        repository=isolated_chat_repository,
    )
    monkeypatch.setattr(chat, "_get_run_manager", lambda: manager)

    response = client.post(
        f"/sessions/{session_id}/execute",
        json={"message": "制定计划"},
    )

    assert response.status_code == 200
    assert "event: todo_status" in response.text
    assert f'"session_id": "{session_id}"' in response.text


def test_run_api_query_sse_replay_and_event_bus_are_unified(
    isolated_chat_repository: ChatSessionRepository,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class ImmediateOrchestrator:
        async def execute_stream(self, message: str, **kwargs: object):
            run_id = str(kwargs["run_id"])
            session_id = str(kwargs["session_id"])
            yield OrchestratorEvent(
                type="final_message",
                run_id=run_id,
                session_id=session_id,
                message="最终答案",
            )
            yield OrchestratorEvent(
                type="done",
                run_id=run_id,
                session_id=session_id,
                message="完成",
            )

    bus = EventBus()
    manager = ChatRunManager(
        orchestrator=ImmediateOrchestrator(),
        repository=isolated_chat_repository,
        bus=bus,
    )
    monkeypatch.setattr(chat, "_get_run_manager", lambda: manager)
    app = FastAPI()
    app.include_router(chat.router)

    with TestClient(app) as client:
        session_id = client.post("/sessions", json={}).json()["session_id"]
        created = client.post(
            f"/sessions/{session_id}/runs",
            json={"message": "执行研究"},
        )
        assert created.status_code == 200
        run_id = created.json()["run_id"]

        queried = client.get(f"/runs/{run_id}")
        assert queried.status_code == 200
        assert queried.json()["run_id"] == run_id

        stream = client.get(f"/runs/{run_id}/events")
        assert stream.status_code == 200
        payloads = [
            json.loads(line.removeprefix("data: "))
            for line in stream.text.splitlines()
            if line.startswith("data: ")
        ]
        assert payloads[0]["sequence"] == 0
        assert [payload["sequence"] for payload in payloads[1:]] == sorted(
            payload["sequence"] for payload in payloads[1:]
        )
        last_sequence = max(payload["sequence"] for payload in payloads)

        reconnect = client.get(
            f"/runs/{run_id}/events?after_sequence={last_sequence}"
        )
        assert reconnect.status_code == 200
        assert "event: snapshot" in reconnect.text
        assert "event: done" not in reconnect.text
        assert [event.payload["sequence"] for event in bus.get_history(run_id)] == list(
            range(1, last_sequence + 1)
        )


def test_cancel_api_returns_cancelling_until_worker_confirms(
    isolated_chat_repository: ChatSessionRepository,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class CancellableOrchestrator:
        async def execute_stream(self, message: str, **kwargs: object):
            cancel_requested = kwargs["cancel_requested"]
            assert callable(cancel_requested)
            while not cancel_requested():
                await asyncio.sleep(0)
            yield OrchestratorEvent(
                type="cancelled",
                run_id=str(kwargs["run_id"]),
                session_id=str(kwargs["session_id"]),
                message="已取消",
            )

    manager = ChatRunManager(
        orchestrator=CancellableOrchestrator(),
        repository=isolated_chat_repository,
    )
    monkeypatch.setattr(chat, "_get_run_manager", lambda: manager)
    app = FastAPI()
    app.include_router(chat.router)

    with TestClient(app) as client:
        session_id = client.post("/sessions", json={}).json()["session_id"]
        run_id = client.post(
            f"/sessions/{session_id}/runs",
            json={"message": "长任务"},
        ).json()["run_id"]
        cancelled = client.post(f"/runs/{run_id}/cancel")
        assert cancelled.status_code == 200
        assert cancelled.json()["status"] == "CANCELLING"

        events = client.get(f"/runs/{run_id}/events")
        assert "event: cancelled" in events.text
        assert client.get(f"/runs/{run_id}").json()["status"] == "CANCELLED"


def test_session_schema_no_mode_field() -> None:
    """ChatSession should not have a required 'mode' field."""
    app = FastAPI()
    app.include_router(chat.router)
    client = TestClient(app)

    session = client.post("/sessions", json={}).json()
    assert "mode" not in session  # No forced mode
    assert "routing_history" not in session
    assert "session_id" in session
    assert "messages" in session


def test_list_sessions_exposes_degraded_storage_status(
    isolated_chat_repository: ChatSessionRepository,
) -> None:
    path = isolated_chat_repository.records.path_for("chat_broken")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{broken", encoding="utf-8")
    app = FastAPI()
    app.include_router(chat.router)
    response = TestClient(app).get("/sessions")
    assert response.status_code == 200
    assert response.headers["X-Storage-Status"] == "DEGRADED"
    assert response.headers["X-Storage-Diagnostics-Count"] == "1"
