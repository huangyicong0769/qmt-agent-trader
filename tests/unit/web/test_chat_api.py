"""Tests for chat API routes — natural language, no forced mode."""

from __future__ import annotations

import anyio
from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.requests import Request

from qmt_agent_trader.agent.orchestrator import OrchestratorEvent
from qmt_agent_trader.web.routes import chat


def test_chat_api_create_and_send_message() -> None:
    chat._sessions.clear()
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
    chat._sessions.clear()
    app = FastAPI()
    app.include_router(chat.router)
    client = TestClient(app)
    client.post("/sessions", json={"title": "One"})

    response = client.get("/sessions")

    assert response.status_code == 200
    assert len(response.json()) == 1


def test_send_message_without_mode_works() -> None:
    """Sending a message without mode should not fail."""
    chat._sessions.clear()
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
    chat._sessions.clear()
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


def test_execute_without_body_reuses_queued_message_without_duplicate() -> None:
    """Executing an enqueued message should not append the same user text again."""
    chat._sessions.clear()
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
    user_messages = [
        message for message in payload["messages"] if message["role"] == "user"
    ]
    assert [message["content"] for message in user_messages] == ["发现因子"]


def test_execute_stream_outputs_todo_status_event_with_session_id(monkeypatch) -> None:
    chat._sessions.clear()
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

    monkeypatch.setattr(chat, "_get_orchestrator", lambda: FakeOrchestrator())

    response = client.post(
        f"/sessions/{session_id}/execute",
        json={"message": "制定计划"},
    )

    assert response.status_code == 200
    assert "event: todo_status" in response.text
    assert f'"session_id": "{session_id}"' in response.text


def test_session_schema_no_mode_field() -> None:
    """ChatSession should not have a required 'mode' field."""
    chat._sessions.clear()
    app = FastAPI()
    app.include_router(chat.router)
    client = TestClient(app)

    session = client.post("/sessions", json={}).json()
    assert "mode" not in session  # No forced mode
    assert "routing_history" not in session
    assert "session_id" in session
    assert "messages" in session
