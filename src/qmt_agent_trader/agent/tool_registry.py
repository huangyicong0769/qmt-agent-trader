"""Tool registry with capability checks."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from qmt_agent_trader.agent.permissions import ToolCapability, assert_llm_tool_allowed


@dataclass(frozen=True)
class ToolDefinition:
    name: str
    capability: ToolCapability
    fn: Callable[..., Any]


@dataclass
class ToolRegistry:
    tools: dict[str, ToolDefinition] = field(default_factory=dict)

    def register(self, definition: ToolDefinition) -> None:
        self.tools[definition.name] = definition

    def call_as_llm(self, name: str, **kwargs: Any) -> Any:
        definition = self.tools[name]
        assert_llm_tool_allowed(definition.capability)
        return definition.fn(**kwargs)
