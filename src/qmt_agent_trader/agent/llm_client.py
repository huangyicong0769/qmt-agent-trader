"""OpenAI-compatible DeepSeek client wrapper."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from openai import OpenAI


@dataclass(frozen=True)
class DeepSeekTool:
    name: str
    description: str
    parameters: dict[str, Any]
    fn: Callable[..., Any]
    strict: bool = False


@dataclass(frozen=True)
class ToolExecution:
    tool_call_id: str
    name: str
    arguments: dict[str, Any]
    result: Any


@dataclass(frozen=True)
class DeepSeekToolLoopResult:
    content: str
    messages: list[dict[str, Any]]
    tool_calls: list[ToolExecution]


class DeepSeekClient:
    def __init__(self, *, api_key: str, base_url: str, model: str) -> None:
        self.model = model
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def complete(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
        )
        content = response.choices[0].message.content
        return content or ""

    def complete_json(self, prompt: str) -> dict[str, Any]:
        content = self.complete(prompt)
        return _parse_json_object(content)

    def run_tool_loop(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[DeepSeekTool],
        max_rounds: int = 4,
    ) -> DeepSeekToolLoopResult:
        if max_rounds < 1:
            raise ValueError("max_rounds must be positive")

        conversation = [dict(message) for message in messages]
        tool_map = {tool.name: tool for tool in tools}
        executions: list[ToolExecution] = []
        request_tools = [_to_openai_tool(tool) for tool in tools]

        for _ in range(max_rounds):
            kwargs: dict[str, Any] = {"model": self.model, "messages": conversation}
            if request_tools:
                kwargs["tools"] = request_tools

            response = self.client.chat.completions.create(**kwargs)
            message = response.choices[0].message
            tool_calls = list(getattr(message, "tool_calls", None) or [])
            conversation.append(_assistant_message_dict(message))

            if not tool_calls:
                return DeepSeekToolLoopResult(
                    content=str(getattr(message, "content", "") or ""),
                    messages=conversation,
                    tool_calls=executions,
                )

            for call in tool_calls:
                name = _function_name(call)
                if name not in tool_map:
                    raise ValueError(f"LLM requested unknown tool: {name}")
                arguments = _function_arguments(call)
                result = tool_map[name].fn(**arguments)
                executions.append(
                    ToolExecution(
                        tool_call_id=str(_attr_or_item(call, "id")),
                        name=name,
                        arguments=arguments,
                        result=result,
                    )
                )
                conversation.append(
                    {
                        "role": "tool",
                        "tool_call_id": str(_attr_or_item(call, "id")),
                        "content": _tool_result_content(result),
                    }
                )

        raise RuntimeError("LLM tool loop exceeded max_rounds")


def _parse_json_object(content: str) -> dict[str, Any]:
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        start = content.find("{")
        end = content.rfind("}")
        if start < 0 or end <= start:
            raise
        parsed = json.loads(content[start : end + 1])
    if not isinstance(parsed, dict):
        raise TypeError("LLM response must be a JSON object")
    return parsed


def _to_openai_tool(tool: DeepSeekTool) -> dict[str, Any]:
    function: dict[str, Any] = {
        "name": tool.name,
        "description": tool.description,
        "parameters": tool.parameters,
    }
    if tool.strict:
        function["strict"] = True
    return {"type": "function", "function": function}


def _assistant_message_dict(message: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {"role": "assistant"}
    content = getattr(message, "content", None)
    if content is not None:
        payload["content"] = content
    tool_calls = list(getattr(message, "tool_calls", None) or [])
    if tool_calls:
        payload["tool_calls"] = [_tool_call_dict(call) for call in tool_calls]
    return payload


def _tool_call_dict(call: Any) -> dict[str, Any]:
    return {
        "id": str(_attr_or_item(call, "id")),
        "type": str(_attr_or_item(call, "type", "function")),
        "function": {
            "name": _function_name(call),
            "arguments": _function_arguments_raw(call),
        },
    }


def _function_name(call: Any) -> str:
    function = _attr_or_item(call, "function")
    return str(_attr_or_item(function, "name"))


def _function_arguments(call: Any) -> dict[str, Any]:
    raw = _function_arguments_raw(call)
    parsed = json.loads(raw or "{}")
    if not isinstance(parsed, dict):
        raise TypeError("tool call arguments must be a JSON object")
    return parsed


def _function_arguments_raw(call: Any) -> str:
    function = _attr_or_item(call, "function")
    return str(_attr_or_item(function, "arguments", "{}") or "{}")


def _tool_result_content(result: Any) -> str:
    if isinstance(result, str):
        return result
    return json.dumps(result, ensure_ascii=False, default=str)


def _attr_or_item(value: Any, key: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(key, default)
    return getattr(value, key, default)
