from __future__ import annotations

import json

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
