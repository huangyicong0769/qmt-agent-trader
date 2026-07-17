"""OpenAI-compatible DeepSeek client wrapper.

Supports both batch (run_tool_loop) and streaming (run_tool_loop_stream)
modes for real-time token delivery and tool call interception.

Loop safety: no hard round limit. Instead, heuristic loop detection stops
the LLM if it repeats the same tool+args 3+ consecutive times.
A generous safety cap (100 rounds) prevents runaway processes.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Callable, Generator
from dataclasses import dataclass
from typing import Any

from openai import OpenAI
from openai.types.chat import (
    ChatCompletionMessageParam,
)

# ── Safety constants ──

_MAX_ROUNDS_SAFETY_CAP = 100
_MAX_REPEAT_BEFORE_BREAK = 3
_MAX_TOOL_RESULT_CONTENT_CHARS = 12_000
_MAX_TOOL_RESULT_LIST_ITEMS = 30
_MAX_TOOL_RESULT_SYMBOL_ITEMS = 500

logger = logging.getLogger(__name__)


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
    phase: str = "draft"


@dataclass(frozen=True)
class FinalMessage(StreamEvent):
    """The final assistant message for the current tool loop."""
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
class LoopBreak(StreamEvent):
    """Heuristic loop detected — same tool+args repeated."""
    message: str


@dataclass(frozen=True)
class SafetyCapHit(StreamEvent):
    """Absolute safety cap (100 rounds) reached."""
    message: str


@dataclass(frozen=True)
class LoopError(StreamEvent):
    """An unexpected error occurred during the loop."""
    message: str


@dataclass(frozen=True)
class Cancelled(StreamEvent):
    """Cooperative cancellation was observed by the tool loop."""

    reason: str = "user_interrupt"


# ── Loop guard ──


def _loop_guard(
    history: list[tuple[str, str]],
    tool_name: str,
    args: dict[str, Any],
    *,
    max_repeat: int = _MAX_REPEAT_BEFORE_BREAK,
) -> str | None:
    """Return a break message if the same (tool_name, args) repeats too many
    consecutive times, otherwise None.

    Args are normalised via stable JSON serialisation to detect same-input
    calls regardless of key ordering.
    """
    args_key = json.dumps(args, sort_keys=True, ensure_ascii=False, default=str)
    entry = (tool_name, args_key)
    history.append(entry)

    if len(history) >= max_repeat:
        if all(e == entry for e in history[-max_repeat:]):
            return (
                f"Loop detected: '{tool_name}' called with the same arguments "
                f"{max_repeat} consecutive times. Stopping."
            )
    return None


# ── Client ──


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
        max_rounds: int = _MAX_ROUNDS_SAFETY_CAP,
    ) -> DeepSeekToolLoopResult:
        """Batch tool loop with heuristic loop detection + safety cap."""
        if max_rounds < 1:
            raise ValueError("max_rounds must be positive")

        conversation = [dict(message) for message in messages]
        tool_map = {tool.name: tool for tool in tools}
        executions: list[ToolExecution] = []
        request_tools = [_to_openai_tool(tool) for tool in tools]
        call_history: list[tuple[str, str]] = []
        force_final_answer = False
        empty_final_answer_retries = 0

        for _ in range(max_rounds):
            kwargs: dict[str, Any] = {"model": self.model, "messages": conversation}
            if request_tools and force_final_answer:
                kwargs["tools"] = request_tools
                kwargs["tool_choice"] = "none"
            elif request_tools:
                kwargs["tools"] = request_tools

            response = self.client.chat.completions.create(**kwargs)
            message = response.choices[0].message
            tool_calls = list(getattr(message, "tool_calls", None) or [])

            assistant_dict = _assistant_message_dict(message)
            if not tool_calls and force_final_answer and assistant_dict.get("content"):
                assistant_dict["content"] = _strip_tool_call_markup(
                    str(assistant_dict["content"])
                ).strip()

            if not tool_calls:
                final_content = str(assistant_dict.get("content", "") or "")
                if force_final_answer and not final_content.strip():
                    assistant_dict.setdefault("content", "")
                    conversation.append(assistant_dict)
                    if empty_final_answer_retries >= 1:
                        raise RuntimeError(
                            "LLM returned an empty final answer after research report generation."
                        )
                    empty_final_answer_retries += 1
                    conversation.append(
                        {
                            "role": "system",
                            "content": (
                                "The previous assistant final answer was empty. Provide a "
                                "concise Chinese final answer now from the observed tool "
                                "evidence and report path. Do not call tools."
                            ),
                        }
                    )
                    continue
                conversation.append(assistant_dict)
                return DeepSeekToolLoopResult(
                    content=final_content,
                    messages=conversation,
                    tool_calls=executions,
                )

            conversation.append(assistant_dict)
            for call in tool_calls:
                name = _function_name(call)
                if name not in tool_map:
                    raise ValueError(f"LLM requested unknown tool: {name}")
                arguments = _function_arguments(call)

                # ── Loop guard ──
                break_msg = _loop_guard(call_history, name, arguments)
                if break_msg:
                    raise RuntimeError(break_msg)

                result = tool_map[name].fn(**arguments)
                if name == "generate_research_report":
                    force_final_answer = True
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
            if force_final_answer:
                conversation.append(
                    {
                        "role": "system",
                        "content": (
                            "Research report has been generated. Do not call more tools; "
                            "do not emit tool-call markup, DSML, JSON function calls, or "
                            "pseudo read_file requests. Answer the user in natural "
                            "language from the observed evidence and report path."
                        ),
                    }
                )

        raise RuntimeError(
            f"Safety cap: LLM tool loop reached {max_rounds} rounds without finishing."
        )

    def run_tool_loop_stream(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[DeepSeekTool],
        max_rounds: int = _MAX_ROUNDS_SAFETY_CAP,
        cancel_requested: Callable[[], bool] | None = None,
    ) -> Generator[StreamEvent, None, None]:
        """Streaming tool loop with heuristic loop detection + safety cap.

        DeepSeek thinking mode: accumulates reasoning_content from stream
        deltas and passes it back in subsequent requests (required by API).
        """
        if max_rounds < 1:
            raise ValueError("max_rounds must be positive")

        conversation: list[ChatCompletionMessageParam] = [
            dict(m) for m in messages  # type: ignore[misc]
        ]
        tool_map = {tool.name: tool for tool in tools}
        request_tools = [_to_openai_tool(tool) for tool in tools]
        call_history: list[tuple[str, str]] = []

        def is_cancel_requested() -> bool:
            return cancel_requested is not None and cancel_requested()

        def cancelled_event() -> Cancelled:
            return Cancelled(reason="user_interrupt")

        for _round in range(max_rounds):
            if is_cancel_requested():
                yield cancelled_event()
                return
            kwargs: dict[str, Any] = {
                "model": self.model,
                "messages": conversation,
                "stream": True,
            }
            if request_tools:
                kwargs["tools"] = request_tools

            if is_cancel_requested():
                yield cancelled_event()
                return
            stream: Any = None
            stream_cancelled = False

            # Accumulators for streaming deltas
            content_parts: list[str] = []
            reasoning_parts: list[str] = []
            tool_call_buf: dict[int, dict[str, Any]] = {}
            finished_tool_calls: list[dict[str, Any]] = []
            remove_cancel_callback: Callable[[], None] | None = None

            try:
                stream = self.client.chat.completions.create(**kwargs)
                register_cancel_callback = getattr(
                    cancel_requested,
                    "add_cancel_callback",
                    None,
                )
                if callable(register_cancel_callback):
                    remove_cancel_callback = register_cancel_callback(
                        lambda stream=stream: _close_stream(stream)
                    )
                if is_cancel_requested():
                    stream_cancelled = True
                else:
                    for chunk in stream:
                        if is_cancel_requested():
                            stream_cancelled = True
                            break
                        delta = chunk.choices[0].delta if chunk.choices else None
                        if delta is None:
                            continue

                        # ── Reasoning content (DeepSeek thinking mode) ──
                        rc = getattr(delta, "reasoning_content", None)
                        if rc:
                            reasoning_parts.append(rc)

                        # ── Text content ──
                        if delta.content:
                            content_parts.append(delta.content)
                            yield TextDelta(content=delta.content, phase="draft")

                        # ── Tool calls (streamed as deltas) ──
                        tc_deltas = list(getattr(delta, "tool_calls", None) or [])
                        for tc in tc_deltas:
                            if is_cancel_requested():
                                stream_cancelled = True
                                break
                            idx = getattr(tc, "index", 0)
                            if idx not in tool_call_buf:
                                tc_id = getattr(tc, "id", "") or ""
                                fn_name = getattr(
                                    getattr(tc, "function", None), "name", ""
                                ) or ""
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
                            new_name = getattr(
                                getattr(tc, "function", None), "name", ""
                            ) or ""
                            if new_name and not buf["name"]:
                                buf["name"] = new_name
                        if stream_cancelled:
                            break
                        if is_cancel_requested():
                            stream_cancelled = True
                            break
            finally:
                if remove_cancel_callback is not None:
                    remove_cancel_callback()
                _close_stream(stream)

            if stream_cancelled or is_cancel_requested():
                yield cancelled_event()
                return

            # ── Build reasoning_content string for conversation ──
            reasoning_str = "".join(reasoning_parts) if reasoning_parts else None

            # ── Complete tool calls: check loop, parse args, execute ──
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

                if is_cancel_requested():
                    yield cancelled_event()
                    return

                # ── Loop guard ──
                break_msg = _loop_guard(call_history, tc_name, arguments)
                if break_msg:
                    yield LoopBreak(message=break_msg)
                    return

                yield ToolCallComplete(
                    tool_call_id=tc_id,
                    tool_name=tc_name,
                    arguments=arguments,
                )

                if is_cancel_requested():
                    yield cancelled_event()
                    return

                if tc_name in tool_map:
                    if is_cancel_requested():
                        yield cancelled_event()
                        return
                    try:
                        result = tool_map[tc_name].fn(**arguments)
                    except Exception as exc:
                        result = {"error": str(exc)}
                    if is_cancel_requested():
                        yield cancelled_event()
                        return
                    yield ToolResult(
                        tool_call_id=tc_id,
                        tool_name=tc_name,
                        result=result,
                    )
                    assistant_msg: dict[str, Any] = {
                        "role": "assistant",
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
                    }
                    if reasoning_str:
                        assistant_msg["reasoning_content"] = reasoning_str
                    conversation.append(assistant_msg)  # type: ignore[arg-type]
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
                    if is_cancel_requested():
                        yield cancelled_event()
                        return
                else:
                    yield LoopError(
                        message=f"LLM requested unknown tool: {tc_name}"
                    )
                    return

            # ── No tool calls? Append final assistant message and done ──
            if not finished_tool_calls:
                if is_cancel_requested():
                    yield cancelled_event()
                    return
                final_content = "".join(content_parts) if content_parts else None
                if final_content:
                    yield FinalMessage(content=final_content)
                if is_cancel_requested():
                    yield cancelled_event()
                    return
                final_msg: dict[str, Any] = {"role": "assistant"}
                if final_content:
                    final_msg["content"] = final_content
                if reasoning_str:
                    final_msg["reasoning_content"] = reasoning_str
                conversation.append(final_msg)  # type: ignore[arg-type]
                return

            if is_cancel_requested():
                yield cancelled_event()
                return

        # Safety cap
        yield SafetyCapHit(
            message=f"Safety cap: LLM tool loop reached {max_rounds} rounds without finishing."
        )


# ── Helpers ──


def _close_stream(stream: Any) -> None:
    if stream is None:
        return
    close = getattr(stream, "close", None)
    if not callable(close):
        return
    try:
        close()
    except Exception as exc:
        logger.debug("model stream close failed during cleanup: %s", exc)


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
    reasoning = getattr(message, "reasoning_content", None)
    if reasoning is not None:
        payload["reasoning_content"] = reasoning
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
    compact = _compact_tool_result(result)
    content = json.dumps(compact, ensure_ascii=False, default=str)
    if len(content) <= _MAX_TOOL_RESULT_CONTENT_CHARS:
        return content
    return json.dumps(
        _fallback_tool_result_summary(result, content),
        ensure_ascii=False,
        default=str,
    )


def _strip_tool_call_markup(content: str) -> str:
    patterns = [
        r"<｜｜DSML｜｜tool_calls>.*?</｜｜DSML｜｜tool_calls>",
        r"<tool_calls>.*?</tool_calls>",
    ]
    cleaned = content
    for pattern in patterns:
        cleaned = re.sub(pattern, "", cleaned, flags=re.DOTALL | re.IGNORECASE)
    return cleaned


def _has_failed_or_incomplete_evidence(messages: list[Any]) -> bool:
    markers = (
        "FAIL",
        "BLOCKED",
        "NOT_COMPUTED",
        "REVIEW_REQUIRED",
        "ADAPTER_LIMITATION",
        "UNSUPPORTED_FORMULA",
    )
    for message in messages:
        if not isinstance(message, dict):
            continue
        if message.get("role") not in {"assistant", "tool"}:
            continue
        content = message.get("content")
        if content is None:
            continue
        text = str(content)
        if any(marker in text for marker in markers):
            return True
    return False


def _neutralize_overclaimed_failed_evidence(
    content: str,
    *,
    has_failed_evidence: bool,
) -> str:
    _ = has_failed_evidence
    return content


def _compact_tool_result(value: Any, *, key: str | None = None) -> Any:
    if isinstance(value, dict):
        return {
            str(item_key): _compact_tool_result(item, key=str(item_key))
            for item_key, item in value.items()
        }
    if isinstance(value, list):
        limit = (
            _MAX_TOOL_RESULT_SYMBOL_ITEMS
            if key in {"symbols", "requested_symbols", "covered_symbols", "missing_symbols"}
            else _MAX_TOOL_RESULT_LIST_ITEMS
        )
        items = [_compact_tool_result(item) for item in value[:limit]]
        omitted = len(value) - len(items)
        if omitted <= 0:
            return items
        return {
            "items": items,
            "omitted_count": omitted,
            "truncated": True,
        }
    return value


def _fallback_tool_result_summary(result: Any, content: str) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "truncated": True,
        "content_chars": len(content),
        "preview": content[:_MAX_TOOL_RESULT_CONTENT_CHARS],
    }
    if isinstance(result, dict):
        for key in (
            "execution_status",
            "domain_status",
            "evidence_status",
            "recommendation_status",
            "raw_status",
            "diagnostic_status",
            "status",
            "reason",
            "message",
            "run_id",
            "strategy_id",
            "factor_id",
            "report_path",
            "code_path",
            "diagnostics",
            "metrics",
            "data_window",
            "coverage_status",
            "missing_symbols",
            "stale_symbols",
            "missing_columns",
            "missing_ranges",
            "data_update_needed",
            "next_repair_tool",
            "suggested_repair",
            "repair_action",
            "verification_action",
            "research_only",
            "review_required",
            "live_trading_allowed",
            "adapter_limitations",
            "data_provenance",
            "column_quality",
            "blockers",
            "warnings",
        ):
            if key in result:
                summary[key] = _compact_tool_result(result[key])
        metadata = result.get("metadata")
        if isinstance(metadata, dict):
            summary["metadata"] = {
                key: _compact_tool_result(metadata[key])
                for key in (
                    "status",
                    "reason",
                    "message",
                    "coverage_status",
                    "missing_symbols",
                    "stale_symbols",
                    "missing_ranges",
                    "data_update_needed",
                    "next_repair_tool",
                    "repair_action",
                    "verification_action",
                )
                if key in metadata
            }
    return summary


def _attr_or_item(value: Any, key: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(key, default)
    return getattr(value, key, default)
