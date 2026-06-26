from __future__ import annotations

from typing import ClassVar

import anyio
from pydantic import SecretStr

from qmt_agent_trader.agent.llm_client import SafetyCapHit, TextDelta
from qmt_agent_trader.agent.orchestrator import AgentOrchestrator
from qmt_agent_trader.core.config import Settings
from qmt_agent_trader.data.storage import DataLake


class CapturingDeepSeekClient:
    seen_messages: ClassVar[list[dict]] = []

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


class SafetyCapDeepSeekClient:
    def __init__(self, **_kwargs: object) -> None:
        pass

    def run_tool_loop_stream(self, *, messages: list[dict], tools: list[object], max_rounds: int):
        yield TextDelta(content="正在评估")
        yield SafetyCapHit(message="Safety cap: LLM tool loop reached 24 rounds without finishing.")


def test_execute_stream_reports_safety_cap_as_error_without_done(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(
        "qmt_agent_trader.agent.orchestrator.DeepSeekClient",
        SafetyCapDeepSeekClient,
    )
    settings = Settings(project_root=tmp_path, deepseek_api_key=SecretStr("key"))
    lake = DataLake(root=tmp_path / "lake", duckdb_path=tmp_path / "db.duckdb")
    orchestrator = AgentOrchestrator(settings=settings, data_lake=lake)

    async def collect_events() -> list[object]:
        return [
            event
            async for event in orchestrator.execute_stream(
                "在这个ETF上寻找有效因子，并进行评估",
                run_id="run-safety-cap",
                max_rounds=24,
            )
        ]

    events = anyio.run(collect_events)

    assert any(event.type == "error" for event in events)
    assert all(event.type != "done" for event in events)
