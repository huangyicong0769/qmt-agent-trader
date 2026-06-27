"""Tests for web tool API permissions."""

from __future__ import annotations

import json

from fastapi import FastAPI
from fastapi.testclient import TestClient

from qmt_agent_trader.agent.audit import AuditLogger
from qmt_agent_trader.agent.permissions import PermissionLevel, ToolCallMode
from qmt_agent_trader.agent.schemas import ToolContext, ToolSpec
from qmt_agent_trader.agent.tool_registry import AgentToolRegistry
from qmt_agent_trader.agent.tools.base import tool
from qmt_agent_trader.web.routes import tools


class _FakeRuntime:
    def __init__(self, registry: AgentToolRegistry) -> None:
        self.registry = registry
        self.seen_contexts: list[ToolContext] = []

    def agent_registry(self) -> AgentToolRegistry:
        return self.registry

    def list_tools(self, *, agent_callable_only: bool = True) -> list[dict[str, object]]:
        return self.registry.list_tools(agent_callable_only=agent_callable_only)

    def describe_tool(self, name: str) -> ToolSpec:
        return self.registry.describe_tool(name)

    def run_tool(
        self,
        name: str,
        input_data: dict[str, object],
        context: ToolContext,
    ) -> dict[str, object]:
        self.seen_contexts.append(context)
        return self.registry.run_tool(name, dict(input_data), context)


def test_tool_api_allows_read_only_and_uses_autonomous_runtime_context(
    tmp_path, monkeypatch
) -> None:
    seen_contexts: list[ToolContext] = []
    registry = AgentToolRegistry(audit_logger=AuditLogger(tmp_path / "audit.jsonl"))
    registry.register(
        tool(
            ToolSpec(name="echo", description="Echo", permission=PermissionLevel.READ_ONLY),
            fn=lambda data, context: seen_contexts.append(context) or {"echo": data},
        )
    )
    runtime = _FakeRuntime(registry)
    monkeypatch.setattr(tools, "get_agent_runtime", lambda: runtime)
    app = FastAPI()
    app.include_router(tools.router)

    response = TestClient(app).post("/echo/run", json={"input_data": {"x": 1}})

    assert response.status_code == 200
    assert response.json()["result"] == {"echo": {"x": 1}}
    assert seen_contexts[0].call_mode == ToolCallMode.AUTONOMOUS_AGENT
    audit = json.loads((tmp_path / "audit.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert audit["call_mode"] == ToolCallMode.AUTONOMOUS_AGENT.value
    assert audit["status"] == "ok"


def test_tool_api_injects_remote_data_update_dry_run_for_web_requests(
    tmp_path, monkeypatch
) -> None:
    seen_inputs: list[dict[str, object]] = []
    registry = AgentToolRegistry(audit_logger=AuditLogger(tmp_path / "audit.jsonl"))
    registry.register(
        tool(
            ToolSpec(
                name="run_remote_data_update",
                description="Update remote data",
                permission=PermissionLevel.RESEARCH_WRITE,
            ),
            fn=lambda data, context: seen_inputs.append(data) or {"status": "planned"},
        )
    )
    monkeypatch.setattr(tools, "get_agent_runtime", lambda: _FakeRuntime(registry))
    app = FastAPI()
    app.include_router(tools.router)

    response = TestClient(app).post(
        "/run_remote_data_update/run",
        json={"input_data": {"start_date": "20240101", "end_date": "20240103"}},
    )

    assert response.status_code == 200
    assert seen_inputs[0]["dry_run"] is True


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
    monkeypatch.setattr(tools, "get_agent_runtime", lambda: _FakeRuntime(registry))
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
    monkeypatch.setattr(tools, "get_agent_runtime", lambda: _FakeRuntime(registry))
    app = FastAPI()
    app.include_router(tools.router)

    response = TestClient(app).post("/forbidden/run", json={"input_data": {}})

    assert response.status_code == 200
    assert response.json()["status"] == "permission_denied"
