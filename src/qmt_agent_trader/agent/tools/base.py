"""Base AgentTool protocol and implementation helpers."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol

from qmt_agent_trader.agent.schemas import ToolContext, ToolSpec


class AgentTool(Protocol):
    """Every tool in the Agent subsystem must satisfy this protocol.

    The `spec` describes the tool; `run` executes it in the given context.
    """

    @property
    def spec(self) -> ToolSpec: ...

    def run(self, input_data: dict[str, Any], context: ToolContext) -> dict[str, Any]:
        ...


# ── Simple tool builder (functional) ──────────────────────────────────────────

_AgentToolFn = Callable[[dict[str, Any], ToolContext], dict[str, Any]]
TimeoutSecondsResolver = Callable[[dict[str, Any], ToolContext], int | float]


class _FnTool:
    """Tool that wraps a plain function."""

    def __init__(
        self,
        spec: ToolSpec,
        fn: _AgentToolFn,
        timeout_seconds_for_call: TimeoutSecondsResolver | None = None,
    ) -> None:
        self._spec = spec
        self._fn = fn
        self._timeout_seconds_for_call = timeout_seconds_for_call

    @property
    def spec(self) -> ToolSpec:
        return self._spec

    def run(self, input_data: dict[str, Any], context: ToolContext) -> dict[str, Any]:
        return self._fn(input_data, context)

    def timeout_seconds_for_call(
        self,
        input_data: dict[str, Any],
        context: ToolContext,
    ) -> int | float | None:
        if self._timeout_seconds_for_call is None:
            return None
        return self._timeout_seconds_for_call(input_data, context)


def tool(
    spec: ToolSpec,
    fn: _AgentToolFn,
    timeout_seconds_for_call: TimeoutSecondsResolver | None = None,
) -> AgentTool:
    """Create a tool from a `ToolSpec` and a plain function."""
    return _FnTool(spec, fn, timeout_seconds_for_call)
