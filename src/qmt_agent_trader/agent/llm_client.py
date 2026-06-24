"""OpenAI-compatible DeepSeek client wrapper.

Supports both batch (run_tool_loop) and streaming (run_tool_loop_stream)
modes for real-time token delivery and tool call interception.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Generator
from dataclasses import dataclass
from typing import Any

from openai import OpenAI
from openai.types.chat import (
    ChatCompletionMessageParam,
)


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


# ── Streaming events ──

@dataclass(frozen=True)
class StreamEvent:
    """A single event emitted during a streaming tool loop."""
    pass


@dataclass(frozen=True)
class TextDelta(StreamEvent):
    """A chunk of LLM text output."""
    content: str


@dataclass(frozen=True)
class ToolCallStart(StreamEvent):
    """LLM is about to call a tool."""
    tool_call_id: str
    tool_name: str


@dataclass(frozen=True)
class ToolCallDelta(StreamEvent):
    """Partial arguments for an in-progress tool call."""
    tool_call_id: str
    arguments_delta: str


@dataclass(frozen=True)
class ToolCallComplete(StreamEvent):
    """Tool call args complete; result follows."""
    tool_call_id: str
    tool_name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class ToolResult(StreamEvent):
    """A tool call executed and produced a result."""
    tool_call_id: str
    tool_name: str
    result: Any


@dataclass(frozen=True)
class LoopError(StreamEvent):
    """An error occurred during the loop."""
    message: str


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

    def run_tool_loop_stream(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[DeepSeekTool],
        max_rounds: int = 6,
    ) -> Generator[StreamEvent, None, None]:
        """Streaming tool loop: yields TextDelta, ToolCallStart/Delta/Complete,
        ToolResult as they happen. Final text is also streamed token-by-token.
        """
        if max_rounds < 1:
            raise ValueError("max_rounds must be positive")

        conversation: list[ChatCompletionMessageParam] = [
            dict(m) for m in messages  # type: ignore[misc]
        ]
        tool_map = {tool.name: tool for tool in tools}
        request_tools = [_to_openai_tool(tool) for tool in tools]

        for _round in range(max_rounds):
            kwargs: dict[str, Any] = {
                "model": self.model,
                "messages": conversation,
                "stream": True,
            }
            if request_tools:
                kwargs["tools"] = request_tools

            stream = self.client.chat.completions.create(**kwargs)

            # Accumulators for streaming deltas
            content_parts: list[str] = []
            tool_call_buf: dict[int, dict[str, Any]] = {}  # index -> {id, name, args_str}
            finished_tool_calls: list[dict[str, Any]] = []

            for chunk in stream:
                delta = chunk.choices[0].delta if chunk.choices else None
                if delta is None:
                    continue

                # ── Text content ──
                if delta.content:
                    content_parts.append(delta.content)
                    yield TextDelta(content=delta.content)

                # ── Tool calls (streamed as deltas) ──
                tc_deltas = list(getattr(delta, "tool_calls", None) or [])
                for tc in tc_deltas:
                    idx = getattr(tc, "index", 0)
                    if idx not in tool_call_buf:
                        tc_id = getattr(tc, "id", "") or ""
                        fn_name = getattr(getattr(tc, "function", None), "name", "") or ""
                        tool_call_buf[idx] = {
                            "id": tc_id,
                            "name": fn_name,
                            "args_str": "",
                        }
                        if fn_name:
                            yield ToolCallStart(
                                tool_call_id=tc_id, tool_name=fn_name
                            )

                    buf = tool_call_buf[idx]
                    fn_delta = getattr(tc, "function", None)
                    if fn_delta:
                        args_d = getattr(fn_delta, "arguments", "") or ""
                        buf["args_str"] += args_d
                        if args_d:
                            yield ToolCallDelta(
                                tool_call_id=buf["id"], arguments_delta=args_d
                            )

                    # If we got a new id/name mid-stream, update
                    new_id = getattr(tc, "id", "") or ""
                    if new_id and not buf["id"]:
                        buf["id"] = new_id
                    new_name = getattr(getattr(tc, "function", None), "name", "") or ""
                    if new_name and not buf["name"]:
                        buf["name"] = new_name

            # ── Complete tool calls: parse args and execute ──
            for idx in sorted(tool_call_buf.keys()):
                buf = tool_call_buf[idx]
                tc_id = buf["id"]
                tc_name = buf["name"]
                args_str = buf["args_str"]

                try:
                    arguments = json.loads(args_str or "{}")
                except json.JSONDecodeError:
                    arguments = {}
                if not isinstance(arguments, dict):
                    arguments = {}

                yield ToolCallComplete(
                    tool_call_id=tc_id,
                    tool_name=tc_name,
                    arguments=arguments,
                )

                if tc_name in tool_map:
                    try:
                        result = tool_map[tc_name].fn(**arguments)
                    except Exception as exc:
                        result = {"error": str(exc)}
                    yield ToolResult(
                        tool_call_id=tc_id,
                        tool_name=tc_name,
                        result=result,
                    )
                    conversation.append({
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": tc_id,
                                "type": "function",
                                "function": {
                                    "name": tc_name,
                                    "arguments": args_str,
                                },
                            }
                        ],
                    })
                    conversation.append({
                        "role": "tool",
                        "tool_call_id": tc_id,
                        "content": _tool_result_content(result),
                    })
                    finished_tool_calls.append({
                        "id": tc_id,
                        "name": tc_name,
                        "arguments": arguments,
                        "result": result,
                    })
                else:
                    yield LoopError(
                        message=f"LLM requested unknown tool: {tc_name}"
                    )
                    return

            # ── No tool calls? Done ──
            if not finished_tool_calls:
                return  # final text already streamed via TextDelta

        # Exhausted max_rounds
        yield LoopError(message="LLM tool loop exceeded max_rounds")


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
