from __future__ import annotations

import anyio
from pydantic import SecretStr

from qmt_agent_trader.agent.orchestrator import AgentOrchestrator
from qmt_agent_trader.core.config import Settings
from qmt_agent_trader.data.storage import DataLake


class CapturingDeepSeekClient:
    seen_messages: list[dict] = []

    def __init__(self, **_kwargs: object) -> None:
        pass

    def run_tool_loop_stream(self, *, messages: list[dict], tools: list[object], max_rounds: int):
        CapturingDeepSeekClient.seen_messages = messages
        return iter(())


def test_execute_stream_includes_recent_natural_session_history(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(
        "qmt_agent_trader.agent.orchestrator.DeepSeekClient",
        CapturingDeepSeekClient,
    )
    settings = Settings(project_root=tmp_path, deepseek_api_key=SecretStr("key"))
    lake = DataLake(root=tmp_path / "lake", duckdb_path=tmp_path / "db.duckdb")
    orchestrator = AgentOrchestrator(settings=settings, data_lake=lake)

    async def collect_events() -> list[object]:
        return [
            event
            async for event in orchestrator.execute_stream(
                "继续回答",
                history=[
                    {"role": "user", "content": "先看 159259"},
                    {"role": "tool", "content": "ignored"},
                    {"role": "assistant", "content": "159259 是 ETF"},
                    {"role": "done", "content": "ignored"},
                    {"role": "user", "content": "继续回答"},
                ],
                run_id="run-test",
            )
        ]

    events = anyio.run(collect_events)

    assert events[-1].type == "done"
    assert CapturingDeepSeekClient.seen_messages[-3:] == [
        {"role": "user", "content": "先看 159259"},
        {"role": "assistant", "content": "159259 是 ETF"},
        {"role": "user", "content": "继续回答"},
    ]
