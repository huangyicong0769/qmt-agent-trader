"""Tool registry with capability checks."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from qmt_agent_trader.agent.llm_client import DeepSeekTool
from qmt_agent_trader.agent.permissions import ToolCapability, assert_llm_tool_allowed


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
