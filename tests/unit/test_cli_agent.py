from __future__ import annotations

import json
from types import SimpleNamespace

from typer.testing import CliRunner

from qmt_agent_trader.cli.main import app


def test_agent_call_tool_uses_agent_tool_registry_for_query_bars() -> None:
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "agent",
            "call-tool",
            "--name",
            "query_bars",
            "--params",
            json.dumps(
                {
                    "symbol": "000001.SZ",
                    "start_date": "20260101",
                    "end_date": "20260105",
                }
            ),
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["tool"] == "query_bars"
    assert "metadata" in payload["result"]


def test_agent_ask_json_preserves_legacy_output_shape(monkeypatch) -> None:
    class FakeRuntime:
        def ask(self, prompt: str, *, max_rounds: int = 100):
            assert prompt == "你好"
            assert max_rounds == 3
            return SimpleNamespace(
                content="回答",
                tool_calls=[
                    SimpleNamespace(
                        name="todo_get_status",
                        arguments={},
                        result={"status": "ok"},
                    )
                ],
            )

    monkeypatch.setattr(
        "qmt_agent_trader.cli.main.build_default_runtime",
        lambda _settings: FakeRuntime(),
    )
    runner = CliRunner()

    result = runner.invoke(
        app,
        ["agent", "ask", "--prompt", "你好", "--max-rounds", "3", "--json"],
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload == {
        "content": "回答",
        "tool_calls": [
            {
                "name": "todo_get_status",
                "arguments": {},
                "result": {"status": "ok"},
            }
        ],
    }
