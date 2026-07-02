from __future__ import annotations

import json
from typing import Any

from qmt_agent_trader.agent.llm_client import (
    DeepSeekClient,
    DeepSeekTool,
    FinalMessage,
    TextDelta,
    ToolResult,
    _neutralize_overclaimed_failed_evidence,
    _parse_json_object,
    _tool_result_content,
)


def test_parse_json_object_accepts_fenced_text() -> None:
    parsed = _parse_json_object('prefix {"a": 1} suffix')
    assert parsed == {"a": 1}


def test_neutralizer_preserves_raw_language_when_failed_evidence_exists() -> None:
    content = "这是最佳候选，显著有效，强烈推荐；收益最高且具有稳健性。"

    neutralized = _neutralize_overclaimed_failed_evidence(
        content,
        has_failed_evidence=True,
    )

    assert neutralized == content


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


class _FakeReportCompletions:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> _Response:
        self.calls.append(kwargs)
        if len(self.calls) == 1:
            return _Response(
                _Message(
                    None,
                    [_ToolCall("call_report", "generate_research_report", {"run_ids": ["r1"]})],
                )
            )
        assert kwargs["tool_choice"] == "none"
        assert kwargs["tools"][0]["function"]["name"] == "generate_research_report"
        assert kwargs["messages"][-1]["role"] == "system"
        return _Response(
            _Message(
                "报告已生成，结论如下。\n"
                "<｜｜DSML｜｜tool_calls><｜｜DSML｜｜invoke name=\"read_file\">"
                "</｜｜DSML｜｜invoke></｜｜DSML｜｜tool_calls>"
            )
        )


class _FakeReportChat:
    def __init__(self) -> None:
        self.completions = _FakeReportCompletions()


class _FakeReportOpenAI:
    def __init__(self) -> None:
        self.chat = _FakeReportChat()


def test_run_tool_loop_forces_final_answer_after_research_report() -> None:
    fake = _FakeReportOpenAI()
    client = DeepSeekClient.__new__(DeepSeekClient)
    client.model = "deepseek-v4-pro"
    client.client = fake

    result = client.run_tool_loop(
        messages=[{"role": "user", "content": "生成报告"}],
        tools=[
            DeepSeekTool(
                name="generate_research_report",
                description="generate report",
                parameters={"type": "object", "properties": {}},
                fn=lambda run_ids: {"report_path": "reports/r.md"},
            )
        ],
    )

    assert result.content == "报告已生成，结论如下。"
    assert "DSML" not in result.messages[-1]["content"]
    assert "read_file" not in result.messages[-1]["content"]
    assert len(fake.chat.completions.calls) == 2
    assert "tools" in fake.chat.completions.calls[0]
    assert "tools" in fake.chat.completions.calls[1]
    assert fake.chat.completions.calls[1]["tool_choice"] == "none"
    final_messages = fake.chat.completions.calls[1]["messages"]
    system_messages = [
        message["content"]
        for message in final_messages
        if message.get("role") == "system"
    ]
    assert any("do not emit tool-call markup" in content for content in system_messages)
    assert any("pseudo read_file requests" in content for content in system_messages)
    assert not any("FAIL/BLOCKED/NOT_COMPUTED" in content for content in system_messages)
    assert not any("repair-required candidates" in content for content in system_messages)


class _FakeEmptyFinalReportCompletions:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> _Response:
        self.calls.append(kwargs)
        if len(self.calls) == 1:
            return _Response(
                _Message(
                    None,
                    [_ToolCall("call_report", "generate_research_report", {"run_ids": ["r1"]})],
                )
            )
        assert kwargs["tool_choice"] == "none"
        assert kwargs["tools"][0]["function"]["name"] == "generate_research_report"
        if len(self.calls) == 2:
            assert kwargs["messages"][-1]["role"] == "system"
            return _Response(_Message(""))
        assert "previous assistant final answer was empty" in kwargs["messages"][-1]["content"]
        return _Response(_Message("最终结论：该策略候选回测失败，需要修复后再评估。"))


class _FakeEmptyFinalReportChat:
    def __init__(self) -> None:
        self.completions = _FakeEmptyFinalReportCompletions()


class _FakeEmptyFinalReportOpenAI:
    def __init__(self) -> None:
        self.chat = _FakeEmptyFinalReportChat()


def test_run_tool_loop_retries_empty_final_answer_after_research_report() -> None:
    fake = _FakeEmptyFinalReportOpenAI()
    client = DeepSeekClient.__new__(DeepSeekClient)
    client.model = "deepseek-v4-pro"
    client.client = fake

    result = client.run_tool_loop(
        messages=[{"role": "user", "content": "生成报告并结论"}],
        tools=[
            DeepSeekTool(
                name="generate_research_report",
                description="generate report",
                parameters={"type": "object", "properties": {}},
                fn=lambda run_ids: {"report_path": "reports/r.md"},
            )
        ],
    )

    assert result.content == "最终结论：该策略候选回测失败，需要修复后再评估。"
    assert len(fake.chat.completions.calls) == 3
    assert "tools" in fake.chat.completions.calls[2]
    assert fake.chat.completions.calls[2]["tool_choice"] == "none"


def test_tool_result_content_compacts_large_lists_for_model_context() -> None:
    content = _tool_result_content(
        {
            "status": "ok",
            "report_path": "reports/research/report.md",
            "factors": [
                {"factor_id": f"factor_{index}", "name": f"factor_{index}"}
                for index in range(200)
            ],
        }
    )
    payload = json.loads(content)

    assert payload["status"] == "ok"
    assert payload["report_path"] == "reports/research/report.md"
    assert payload["factors"]["truncated"] is True
    assert payload["factors"]["omitted_count"] == 170
    assert len(payload["factors"]["items"]) == 30


def test_tool_result_content_preserves_symbol_contract_lists() -> None:
    content = _tool_result_content(
        {
            "status": "OK",
            "symbols": [f"{index:06d}.SZ" for index in range(119)],
            "metadata": {"count": 119},
        }
    )
    payload = json.loads(content)

    assert len(payload["symbols"]) == 119
    assert payload["symbols"][0] == "000000.SZ"
    assert payload["symbols"][-1] == "000118.SZ"


class _StreamFunctionDelta:
    def __init__(self, name: str = "", arguments: str = "") -> None:
        self.name = name
        self.arguments = arguments


class _StreamToolCallDelta:
    def __init__(
        self,
        *,
        index: int,
        tool_call_id: str = "",
        name: str = "",
        arguments: str = "",
    ) -> None:
        self.index = index
        self.id = tool_call_id
        self.function = _StreamFunctionDelta(name=name, arguments=arguments)


class _StreamDelta:
    def __init__(
        self,
        *,
        content: str | None = None,
        tool_calls: list[_StreamToolCallDelta] | None = None,
    ) -> None:
        self.content = content
        self.tool_calls = tool_calls


class _StreamChoice:
    def __init__(self, delta: _StreamDelta) -> None:
        self.delta = delta


class _StreamChunk:
    def __init__(self, delta: _StreamDelta) -> None:
        self.choices = [_StreamChoice(delta)]


class _FakeStreamingCompletions:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if len(self.calls) == 1:
            return iter(
                [
                    _StreamChunk(_StreamDelta(content="草稿：本地数据过期。")),
                    _StreamChunk(
                        _StreamDelta(
                            tool_calls=[
                                _StreamToolCallDelta(
                                    index=0,
                                    tool_call_id="call_1",
                                    name="lookup",
                                    arguments='{"symbol":"000001.SZ"}',
                                )
                            ]
                        )
                    ),
                ]
            )
        return iter([_StreamChunk(_StreamDelta(content="最终：工具结果正常。"))])


class _FakeStreamingChat:
    def __init__(self) -> None:
        self.completions = _FakeStreamingCompletions()


class _FakeStreamingOpenAI:
    def __init__(self) -> None:
        self.chat = _FakeStreamingChat()


def test_run_tool_loop_stream_marks_draft_text_and_emits_final_message() -> None:
    fake = _FakeStreamingOpenAI()
    client = DeepSeekClient.__new__(DeepSeekClient)
    client.model = "deepseek-v4-pro"
    client.client = fake

    events = list(
        client.run_tool_loop_stream(
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
    )

    text_events = [event for event in events if isinstance(event, TextDelta)]
    assert [(event.content, event.phase) for event in text_events] == [
        ("草稿：本地数据过期。", "draft"),
        ("最终：工具结果正常。", "draft"),
    ]
    final_events = [event.content for event in events if isinstance(event, FinalMessage)]
    assert final_events == ["最终：工具结果正常。"]
    assert any(isinstance(event, ToolResult) for event in events)
