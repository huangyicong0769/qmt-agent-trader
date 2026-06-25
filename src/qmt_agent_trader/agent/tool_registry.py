"""Tool registry with capability and permission checks.

This module provides two registries:
1.  `ToolRegistry` (original): DeepSeek-tool-compatible, based on `ToolCapability`.
2.  `AgentToolRegistry` (new): based on `AgentTool` protocol and `PermissionLevel`.

Both coexist; `AgentToolRegistry` delegates LLM-tool generation to the original
registry when needed.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from qmt_agent_trader.agent.audit import AuditLogger
from qmt_agent_trader.agent.errors import ToolDuplicateError, ToolExecutionError, ToolNotFoundError
from qmt_agent_trader.agent.llm_client import DeepSeekTool
from qmt_agent_trader.agent.permissions import (
    ToolCapability,
    assert_llm_tool_allowed,
    can_llm_call,
    require_permission,
    to_capability,
)
from qmt_agent_trader.agent.schemas import ToolContext, ToolSpec
from qmt_agent_trader.agent.tools.base import AgentTool

# ── Original ToolDefinition + ToolRegistry (preserved) ───────────────────────


@dataclass(frozen=True)
class ToolDefinition:
    name: str
    capability: ToolCapability
    fn: Callable[..., Any]
    description: str = ""
    parameters: dict[str, Any] = field(
        default_factory=lambda: {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        }
    )

    def as_deepseek_tool(self) -> DeepSeekTool:
        def guarded_fn(**kwargs: Any) -> Any:
            assert_llm_tool_allowed(self.capability)
            return self.fn(**kwargs)

        return DeepSeekTool(
            name=self.name,
            description=self.description or self.name,
            parameters=self.parameters,
            fn=guarded_fn,
        )


@dataclass
class ToolRegistry:
    tools: dict[str, ToolDefinition] = field(default_factory=dict)

    def register(self, definition: ToolDefinition) -> None:
        self.tools[definition.name] = definition

    def list_tools(self) -> list[dict[str, object]]:
        return [
            {
                "name": definition.name,
                "capability": definition.capability.value,
                "description": definition.description,
                "parameters": definition.parameters,
            }
            for definition in sorted(self.tools.values(), key=lambda item: item.name)
        ]

    def call_as_llm(self, tool_name: str, **kwargs: Any) -> Any:
        definition = self.tools[tool_name]
        assert_llm_tool_allowed(definition.capability)
        return definition.fn(**kwargs)

    def deepseek_tools_for_llm(self) -> list[DeepSeekTool]:
        tools: list[DeepSeekTool] = []
        for definition in sorted(self.tools.values(), key=lambda item: item.name):
            assert_llm_tool_allowed(definition.capability)
            tools.append(definition.as_deepseek_tool())
        return tools


# ── New AgentToolRegistry ────────────────────────────────────────────────────


@dataclass
class AgentToolRegistry:
    """Tool registry built around the `AgentTool` protocol and `PermissionLevel`.

    Every invocation is permission-checked and audit-logged.
    """

    tools: dict[str, AgentTool] = field(default_factory=dict)
    audit_logger: AuditLogger | None = None
    _audit_path: Path | None = None

    # ── Registration ──────────────────────────────────────────────────────

    def register(self, tool: AgentTool) -> None:
        name = tool.spec.name
        if name in self.tools:
            raise ToolDuplicateError(f"tool '{name}' is already registered")
        self.tools[name] = tool

    def register_all(self, *tools: AgentTool) -> None:
        for entry in tools:
            self.register(entry)

    # ── Discovery ─────────────────────────────────────────────────────────

    def list_tools(
        self, *, permission: str | None = None
    ) -> list[dict[str, object]]:
        result: list[dict[str, object]] = []
        for _name, tool in sorted(self.tools.items()):
            spec = tool.spec
            if permission is not None and spec.permission.value != permission:
                continue
            result.append(
                {
                    "name": spec.name,
                    "description": spec.description,
                    "permission": spec.permission.value,
                    "input_schema": spec.input_schema,
                    "output_schema": spec.output_schema,
                    "side_effect_level": spec.side_effect_level,
                    "deterministic": spec.deterministic,
                    "llm_callable": can_llm_call(spec.permission),
                }
            )
        return result

    def describe_tool(self, name: str) -> ToolSpec:
        tool = self._require_tool(name)
        return tool.spec

    # ── Execution ─────────────────────────────────────────────────────────

    def run_tool(
        self,
        name: str,
        input_data: dict[str, Any],
        context: ToolContext,
    ) -> dict[str, Any]:
        tool = self._require_tool(name)
        spec = tool.spec

        # 1. Permissions
        require_permission(
            spec.permission,
            requested_by_llm=context.requested_by_llm,
            tool_name=name,
        )

        # 2. Audit (before)
        start_ms = int(time.monotonic() * 1000)

        # 3. Execute
        status = "ok"
        error_message = None
        result: dict[str, Any] = {}
        try:
            result = tool.run(input_data, context)
            if not isinstance(result, dict):
                result = {"value": result}
        except Exception as exc:
            status = "permission_denied" if "PermissionDenied" in type(exc).__name__ else "error"
            error_message = str(exc)
            result = {"error": True, "message": error_message}

        # 4. Audit (after)
        duration_ms = int(time.monotonic() * 1000) - start_ms
        self._audit_entry(
            tool_name=name,
            run_id=context.run_id,
            experiment_id=context.experiment_id,
            permission=spec.permission.value,
            requested_by_llm=context.requested_by_llm,
            input_data=input_data,
            output_data=result,
            status=status,
            error_message=error_message,
            duration_ms=duration_ms,
        )

        if error_message is not None:
            raise ToolExecutionError(name, Exception(error_message))

        return result

    # ── Bridge: expose as original ToolRegistry (for LLM client) ──────────

    def to_legacy_registry(
        self,
        *,
        context_factory: Callable[[], ToolContext] | None = None,
        llm_callable_only: bool = True,
    ) -> ToolRegistry:
        legacy = ToolRegistry()
        for name, tool in sorted(self.tools.items()):
            spec = tool.spec
            if llm_callable_only and not can_llm_call(spec.permission):
                continue
            capability = to_capability(spec.permission)

            def build_fn(nt: str) -> Callable[..., Any]:
                def fn(**kwargs: Any) -> dict[str, Any]:
                    context = (
                        context_factory()
                        if context_factory is not None
                        else ToolContext(run_id="legacy")
                    )
                    return self.run_tool(nt, kwargs, context)

                return fn

            legacy.register(
                ToolDefinition(
                    name=name,
                    capability=capability,
                    fn=build_fn(name),
                    description=spec.description,
                    parameters=_llm_input_schema(spec.input_schema),
                )
            )
        return legacy

    # ── Internal helpers ──────────────────────────────────────────────────

    def _require_tool(self, name: str) -> AgentTool:
        if name not in self.tools:
            raise ToolNotFoundError(f"tool '{name}' is not registered")
        return self.tools[name]

    def _audit_entry(
        self,
        *,
        tool_name: str,
        run_id: str,
        experiment_id: str | None,
        permission: str,
        requested_by_llm: bool,
        input_data: dict[str, Any] | None,
        output_data: dict[str, Any] | None,
        status: str,
        error_message: str | None,
        duration_ms: int,
    ) -> None:
        if self.audit_logger is not None:
            try:
                self.audit_logger.append(
                    tool_name=tool_name,
                    run_id=run_id,
                    experiment_id=experiment_id,
                    permission=permission,
                    requested_by_llm=requested_by_llm,
                    input_data=input_data,
                    output_data=output_data,
                    status=status,
                    error_message=error_message,
                    duration_ms=duration_ms,
                )
            except Exception:
                pass  # audit failure must not break tool execution


def _llm_input_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Return an OpenAI function-tool compatible object schema."""
    if schema.get("type") == "object":
        return schema
    if not schema:
        return {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        }
    return {
        "type": "object",
        "properties": schema.get("properties", {}),
        "required": schema.get("required", []),
        "additionalProperties": schema.get("additionalProperties", False),
    }
