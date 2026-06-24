"""Tests for chat API routes — natural language, no forced mode."""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient

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

    # Should return a routing decision
    rd = payload["routing_decision"]
    assert "intent" in rd
    assert "confidence" in rd
    assert "rationale" in rd

    # The message should be in the session
    assert "message" in payload
    assert "session_id" in payload
    assert "run_id" in payload


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
    assert payload["routing_decision"]["intent"] == "BACKTEST_ANALYSIS"


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


def test_session_schema_no_mode_field() -> None:
    """ChatSession should not have a required 'mode' field."""
    chat._sessions.clear()
    app = FastAPI()
    app.include_router(chat.router)
    client = TestClient(app)

    session = client.post("/sessions", json={}).json()
    assert "mode" not in session  # No forced mode
    assert "session_id" in session
    assert "messages" in session
