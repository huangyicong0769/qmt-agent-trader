from __future__ import annotations

import json
from typing import Any

from qmt_agent_trader.agent.llm_client import DeepSeekClient, DeepSeekTool, _parse_json_object


def test_parse_json_object_accepts_fenced_text() -> None:
    parsed = _parse_json_object('prefix {"a": 1} suffix')
    assert parsed == {"a": 1}


class _Function:
    def __init__(self, name: str, arguments: dict[str, Any]) -> None:
        self.name = name
        self.arguments = json.dumps(arguments)


class _ToolCall:
    type = "function"

    def __init__(self, tool_call_id: str, name: str, arguments: dict[str, Any]) -> None:
        self.id = tool_call_id
        self.function = _Function(name, arguments)


class _Message:
    def __init__(self, content: str | None, tool_calls: list[_ToolCall] | None = None) -> None:
        self.content = content
        self.tool_calls = tool_calls


class _Choice:
    def __init__(self, message: _Message) -> None:
        self.message = message


class _Response:
    def __init__(self, message: _Message) -> None:
        self.choices = [_Choice(message)]


class _FakeCompletions:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> _Response:
        self.calls.append(kwargs)
        if len(self.calls) == 1:
            return _Response(
                _Message(None, [_ToolCall("call_1", "lookup", {"symbol": "000001.SZ"})])
            )
        assert kwargs["messages"][-1]["role"] == "tool"
        assert kwargs["messages"][-1]["tool_call_id"] == "call_1"
        return _Response(_Message('{"ok": true, "symbol": "000001.SZ"}'))


class _FakeChat:
    def __init__(self) -> None:
        self.completions = _FakeCompletions()


class _FakeOpenAI:
    def __init__(self) -> None:
        self.chat = _FakeChat()


def test_run_tool_loop_executes_deepseek_function_call() -> None:
    fake = _FakeOpenAI()
    client = DeepSeekClient.__new__(DeepSeekClient)
    client.model = "deepseek-v4-pro"
    client.client = fake

    result = client.run_tool_loop(
        messages=[{"role": "user", "content": "lookup first"}],
        tools=[
            DeepSeekTool(
                name="lookup",
                description="lookup a symbol",
                parameters={
                    "type": "object",
                    "properties": {"symbol": {"type": "string"}},
                    "required": ["symbol"],
                    "additionalProperties": False,
                },
                fn=lambda symbol: {"symbol": symbol, "price": 10.0},
            )
        ],
    )

    assert _parse_json_object(result.content)["ok"] is True
    assert result.tool_calls[0].name == "lookup"
    assert result.tool_calls[0].arguments == {"symbol": "000001.SZ"}
    assert result.messages[-2]["role"] == "tool"
    assert fake.chat.completions.calls[0]["tools"][0]["function"]["name"] == "lookup"
