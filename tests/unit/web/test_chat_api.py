"""Tests for chat API routes."""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient

from qmt_agent_trader.web.routes import chat


def test_chat_api_create_and_send_message() -> None:
    chat._sessions.clear()
    app = FastAPI()
    app.include_router(chat.router)
    client = TestClient(app)

    created = client.post("/sessions", json={"title": "Research", "mode": "factor"}).json()
    session_id = created["session_id"]
    response = client.post(f"/sessions/{session_id}/messages", json={"content": "hello"})

    assert response.status_code == 200
    payload = response.json()
    assert len(payload["messages"]) == 2
    assert payload["messages"][0]["role"] == "user"
    assert payload["messages"][1]["metadata"]["stub"] is True


def test_chat_api_lists_sessions() -> None:
    chat._sessions.clear()
    app = FastAPI()
    app.include_router(chat.router)
    client = TestClient(app)
    client.post("/sessions", json={"title": "One"})

    response = client.get("/sessions")

    assert response.status_code == 200
    assert len(response.json()) == 1
