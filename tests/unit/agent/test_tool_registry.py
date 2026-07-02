"""Tests for AgentToolRegistry and legacy ToolRegistry."""

from __future__ import annotations

import json
import time

import pytest

from qmt_agent_trader.agent.audit import AuditLogger
from qmt_agent_trader.agent.errors import (
    ToolDuplicateError,
    ToolExecutionError,
    ToolNotFoundError,
)
from qmt_agent_trader.agent.permissions import PermissionLevel, ToolCapability
from qmt_agent_trader.agent.schemas import ToolContext, ToolSpec
from qmt_agent_trader.agent.tool_registry import AgentToolRegistry, ToolDefinition, ToolRegistry
from qmt_agent_trader.agent.tools.base import AgentTool, tool
from qmt_agent_trader.core.errors import PermissionDeniedError

# ── Helpers ──────────────────────────────────────────────────────────────────


def _echo_tool(name: str, *, permission: PermissionLevel = PermissionLevel.READ_ONLY) -> AgentTool:
    return tool(
        ToolSpec(
            name=name,
            description=f"Echo tool: {name}",
            permission=permission,
            deterministic=True,
        ),
        fn=lambda data, ctx: {"echo": data, "run_id": ctx.run_id},
    )


def _registry(*names: str) -> AgentToolRegistry:
    reg = AgentToolRegistry()
    for name in names:
        reg.register(_echo_tool(name))
    return reg


# ── Registration ─────────────────────────────────────────────────────────────


def test_register_tool() -> None:
    reg = AgentToolRegistry()
    reg.register(_echo_tool("hello"))
    assert len(reg.tools) == 1


def test_register_duplicate_raises() -> None:
    reg = _registry("dup")
    with pytest.raises(ToolDuplicateError):
        reg.register(_echo_tool("dup"))


def test_list_tools() -> None:
    reg = _registry("a", "b")
    listed = reg.list_tools()
    assert len(listed) == 2
    names = {item["name"] for item in listed}
    assert names == {"a", "b"}


def test_list_tools_filter_by_permission() -> None:
    reg = AgentToolRegistry()
    reg.register(_echo_tool("ro", permission=PermissionLevel.READ_ONLY))
    reg.register(_echo_tool("rw", permission=PermissionLevel.RESEARCH_WRITE))
    listed = reg.list_tools(permission="READ_ONLY")
    assert len(listed) == 1
    assert listed[0]["name"] == "ro"


def test_describe_tool() -> None:
    reg = _registry("hello")
    spec = reg.describe_tool("hello")
    assert spec.name == "hello"


def test_describe_missing_raises() -> None:
    reg = AgentToolRegistry()
    with pytest.raises(ToolNotFoundError):
        reg.describe_tool("nope")


# ── Execution — happy path ───────────────────────────────────────────────────


def test_run_tool_success() -> None:
    reg = _registry("echo")
    result = reg.run_tool("echo", {"msg": "hi"}, ToolContext(run_id="r1"))
    assert result["echo"] == {"msg": "hi"}
    assert result["run_id"] == "r1"


def test_run_tool_with_audit(tmp_path) -> None:
    audit = AuditLogger(tmp_path / "audit.jsonl")
    reg = _registry("echo")
    reg.audit_logger = audit
    reg.run_tool("echo", {"x": 1}, ToolContext(run_id="r2"))
    assert audit.log_path.exists()
    lines = audit.log_path.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 1
    entry = json.loads(lines[0])
    assert entry["output_data"]["echo"] == {"x": 1}
    assert entry["output_data"]["run_id"] == "r2"
    assert entry["execution_status"] == "OK"
    assert entry["domain_status"] == "UNKNOWN"
    assert entry["evidence_status"] == "UNKNOWN"


def test_run_tool_writes_started_audit_for_llm_calls(tmp_path) -> None:
    audit = AuditLogger(tmp_path / "audit.jsonl")
    reg = _registry("echo")
    reg.audit_logger = audit

    reg.run_tool(
        "echo",
        {"x": 1},
        ToolContext(
            run_id="r-started",
            requested_by_llm=True,
            session_id="session-started",
        ),
    )

    entries = [
        json.loads(line)
        for line in audit.log_path.read_text(encoding="utf-8").strip().split("\n")
    ]
    assert [entry["status"] for entry in entries] == [
        "started",
        "execution_ok_domain_unknown",
    ]
    assert entries[0]["output_data"]["status"] == "STARTED"
    assert entries[0]["output_data"]["tool_name"] == "echo"
    assert entries[0]["output_data"]["timeout_seconds"] == 60
    assert entries[0]["execution_status"] == "STARTED"


# ── Execution — permissions ──────────────────────────────────────────────────


def test_run_forbidden_tool_raises() -> None:
    reg = AgentToolRegistry()
    reg.register(_echo_tool("forbidden", permission=PermissionLevel.FORBIDDEN_TO_LLM))
    with pytest.raises(PermissionDeniedError):
        reg.run_tool("forbidden", {}, ToolContext(run_id="r3"))


def test_run_approval_required_by_llm_raises() -> None:
    reg = AgentToolRegistry()
    reg.register(
        _echo_tool("needs_human", permission=PermissionLevel.APPROVAL_REQUIRED)
    )
    with pytest.raises(PermissionDeniedError):
        reg.run_tool(
            "needs_human",
            {},
            ToolContext(run_id="r4", requested_by_llm=True),
        )


# ── Execution — error handling ───────────────────────────────────────────────


def test_run_tool_error_is_audited(tmp_path) -> None:
    audit = AuditLogger(tmp_path / "audit.jsonl")
    reg = AgentToolRegistry()
    reg.audit_logger = audit

    def _fail(_d: dict, _c: ToolContext) -> dict:
        raise ValueError("bang")

    reg.register(
        tool(
            ToolSpec(name="fragile", description="fragile", permission=PermissionLevel.READ_ONLY),
            fn=_fail,
        )
    )
    with pytest.raises(ToolExecutionError):
        reg.run_tool("fragile", {}, ToolContext(run_id="r5"))
    assert audit.log_path.exists()


def test_run_tool_timeout_returns_structured_result_and_audits(tmp_path) -> None:
    audit = AuditLogger(tmp_path / "audit.jsonl")
    reg = AgentToolRegistry(audit_logger=audit)

    def _slow(_d: dict, _c: ToolContext) -> dict:
        time.sleep(0.2)
        return {"late": True}

    reg.register(
        tool(
            ToolSpec(
                name="slow",
                description="slow",
                permission=PermissionLevel.READ_ONLY,
                timeout_seconds=0,
            ),
            fn=_slow,
        )
    )

    result = reg.run_tool("slow", {}, ToolContext(run_id="r-timeout"))

    assert result["status"] == "TIMEOUT"
    assert result["tool_name"] == "slow"
    assert result["timeout_seconds"] == 0
    assert result["kill_attempted"] is True
    assert isinstance(result["duration_ms"], int)
    assert '"status": "execution_timeout"' in audit.log_path.read_text(encoding="utf-8")


def test_run_tool_uses_dynamic_timeout_resolver() -> None:
    reg = AgentToolRegistry()

    def _slow(_d: dict, _c: ToolContext) -> dict:
        time.sleep(0.02)
        return {"finished": True}

    reg.register(
        tool(
            ToolSpec(
                name="dynamic_timeout",
                description="dynamic_timeout",
                permission=PermissionLevel.READ_ONLY,
                timeout_seconds=0,
            ),
            fn=_slow,
            timeout_seconds_for_call=lambda _data, _ctx: 1,
        )
    )

    result = reg.run_tool("dynamic_timeout", {}, ToolContext(run_id="r-dynamic-timeout"))

    assert result["finished"] is True
    assert result["execution_status"] == "OK"
    assert result["domain_status"] == "UNKNOWN"


def test_run_tool_timeout_reports_dynamic_timeout_used(tmp_path) -> None:
    audit = AuditLogger(tmp_path / "audit.jsonl")
    reg = AgentToolRegistry(audit_logger=audit)

    def _slow(_d: dict, _c: ToolContext) -> dict:
        time.sleep(0.2)
        return {"late": True}

    reg.register(
        tool(
            ToolSpec(
                name="dynamic_timeout_slow",
                description="dynamic_timeout_slow",
                permission=PermissionLevel.READ_ONLY,
                timeout_seconds=1,
            ),
            fn=_slow,
            timeout_seconds_for_call=lambda _data, _ctx: 0,
        )
    )

    result = reg.run_tool(
        "dynamic_timeout_slow",
        {},
        ToolContext(run_id="r-dynamic-timeout-result"),
    )

    assert result["status"] == "TIMEOUT"
    assert result["tool_name"] == "dynamic_timeout_slow"
    assert result["timeout_seconds"] == 0
    assert result["kill_attempted"] is True
    audit_entry = json.loads(audit.log_path.read_text(encoding="utf-8").strip())
    assert audit_entry["output_data"]["timeout_seconds"] == 0


# ── Legacy ToolRegistry ──────────────────────────────────────────────────────


def test_legacy_register_and_list() -> None:
    reg = ToolRegistry()
    called: list[str] = []

    def my_fn(x: str = "0") -> str:
        called.append(x)
        return x

    reg.register(
        ToolDefinition(
            name="test",
            capability=ToolCapability.READ_DATA,
            fn=my_fn,
            parameters={
                "type": "object",
                "properties": {"x": {"type": "string"}},
            },
        )
    )
    assert len(reg.list_tools()) == 1


def test_agent_registry_legacy_bridge_exports_llm_callable_tools_only() -> None:
    reg = AgentToolRegistry()
    reg.register(_echo_tool("read", permission=PermissionLevel.READ_ONLY))
    reg.register(_echo_tool("code", permission=PermissionLevel.CODE_GENERATION))
    reg.register(_echo_tool("backtest", permission=PermissionLevel.BACKTEST_EXECUTE))
    reg.register(_echo_tool("approval", permission=PermissionLevel.APPROVAL_REQUIRED))
    reg.register(_echo_tool("forbidden", permission=PermissionLevel.FORBIDDEN_TO_LLM))
    reg.register(
        tool(
            ToolSpec(
                name="hidden_read",
                description="registered but hidden from LLM",
                permission=PermissionLevel.READ_ONLY,
                llm_callable=False,
            ),
            fn=lambda data, context: {"ok": True},
        )
    )

    legacy = reg.to_legacy_registry()
    names = set(legacy.tools)

    assert {"read", "code", "backtest"}.issubset(names)
    assert "approval" not in names
    assert "forbidden" not in names
    assert "hidden_read" not in names
    deepseek_tools = legacy.deepseek_tools_for_llm()
    assert {item.name for item in deepseek_tools} == names
    assert all(item.parameters.get("type") == "object" for item in deepseek_tools)


def test_agent_registry_legacy_bridge_uses_context_factory() -> None:
    seen_contexts: list[ToolContext] = []
    reg = AgentToolRegistry()
    reg.register(
        tool(
            ToolSpec(name="echo_context", description="Echo context"),
            fn=lambda data, context: seen_contexts.append(context) or {
                "run_id": context.run_id,
                "experiment_id": context.experiment_id,
            },
        )
    )

    legacy = reg.to_legacy_registry(
        context_factory=lambda: ToolContext(run_id="run-llm", experiment_id="exp-1")
    )

    result = legacy.call_as_llm("echo_context", value=1)

    assert result["run_id"] == "run-llm"
    assert result["experiment_id"] == "exp-1"
    assert result["domain_status"] == "UNKNOWN"
    assert seen_contexts[0].requested_by_llm is True
