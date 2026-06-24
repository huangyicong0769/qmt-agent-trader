"""Tests for web tool API permissions."""

from __future__ import annotations

import json

from fastapi import FastAPI
from fastapi.testclient import TestClient

from qmt_agent_trader.agent.audit import AuditLogger
from qmt_agent_trader.agent.permissions import PermissionLevel
from qmt_agent_trader.agent.schemas import ToolContext, ToolSpec
from qmt_agent_trader.agent.tool_registry import AgentToolRegistry
from qmt_agent_trader.agent.tools.base import tool
from qmt_agent_trader.web.routes import tools


def test_tool_api_allows_read_only_and_sets_human_context(tmp_path, monkeypatch) -> None:
    seen_contexts: list[ToolContext] = []
    registry = AgentToolRegistry(audit_logger=AuditLogger(tmp_path / "audit.jsonl"))
    registry.register(
        tool(
            ToolSpec(name="echo", description="Echo", permission=PermissionLevel.READ_ONLY),
            fn=lambda data, context: seen_contexts.append(context) or {"echo": data},
        )
    )
    monkeypatch.setattr(tools, "get_registry", lambda: registry)
    app = FastAPI()
    app.include_router(tools.router)

    response = TestClient(app).post("/echo/run", json={"input_data": {"x": 1}})

    assert response.status_code == 200
    assert response.json()["result"] == {"echo": {"x": 1}}
    assert seen_contexts[0].requested_by_llm is False
    audit = json.loads((tmp_path / "audit.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert audit["requested_by_llm"] is False
    assert audit["status"] == "ok"


def test_tool_api_blocks_approval_required_and_audits(tmp_path, monkeypatch) -> None:
    registry = AgentToolRegistry(audit_logger=AuditLogger(tmp_path / "audit.jsonl"))
    registry.register(
        tool(
            ToolSpec(
                name="approval",
                description="Needs approval",
                permission=PermissionLevel.APPROVAL_REQUIRED,
            ),
            fn=lambda data, context: {"should_not": "run"},
        )
    )
    monkeypatch.setattr(tools, "get_registry", lambda: registry)
    app = FastAPI()
    app.include_router(tools.router)

    response = TestClient(app).post("/approval/run", json={"input_data": {}})

    assert response.status_code == 200
    assert response.json()["status"] == "permission_denied"
    audit = json.loads((tmp_path / "audit.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert audit["status"] == "permission_denied"


def test_tool_api_blocks_forbidden_tools(tmp_path, monkeypatch) -> None:
    registry = AgentToolRegistry(audit_logger=AuditLogger(tmp_path / "audit.jsonl"))
    registry.register(
        tool(
            ToolSpec(
                name="forbidden",
                description="Forbidden",
                permission=PermissionLevel.FORBIDDEN_TO_LLM,
            ),
            fn=lambda data, context: {"should_not": "run"},
        )
    )
    monkeypatch.setattr(tools, "get_registry", lambda: registry)
    app = FastAPI()
    app.include_router(tools.router)

    response = TestClient(app).post("/forbidden/run", json={"input_data": {}})

    assert response.status_code == 200
    assert response.json()["status"] == "permission_denied"
