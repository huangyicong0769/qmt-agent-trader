from __future__ import annotations

import json
from types import SimpleNamespace

from typer.testing import CliRunner

from qmt_agent_trader.cli.main import app
from qmt_agent_trader.core.types import ApprovalStatus
from qmt_agent_trader.strategy.models import SavedStrategy, StrategySource, StrategySpec


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


def test_strategy_approve_rejects_non_review_required_candidate(monkeypatch) -> None:
    saved = SavedStrategy(
        strategy_id="strat_draft",
        name="draft",
        version="0.1.0",
        source=StrategySource.AGENT_GENERATED,
        status=ApprovalStatus.GENERATED_BY_LLM,
        spec=StrategySpec(strategy_id="strat_draft", name="draft"),
        implementation_ref="file:strategy.py",
    )

    class FakeRegistry:
        def get_strategy(self, strategy_id: str) -> SavedStrategy | None:
            return saved if strategy_id == "strat_draft" else None

    monkeypatch.setattr("qmt_agent_trader.cli.main._strategy_registry", lambda: FakeRegistry())

    result = CliRunner().invoke(app, ["strategy", "approve", "--strategy-id", "strat_draft"])

    assert result.exit_code != 0
    assert "REVIEW_REQUIRED" in result.output
