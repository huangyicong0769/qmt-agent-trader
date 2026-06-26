"""Regression tests for chat orchestrator tool exposure."""

from __future__ import annotations

from qmt_agent_trader.agent.orchestrator import AgentOrchestrator
from qmt_agent_trader.core.config import Settings


def test_orchestrator_exposes_full_agent_tool_registry_to_llm(tmp_path) -> None:
    orchestrator = AgentOrchestrator(
        settings=Settings(
            project_root=tmp_path,
            qmt_gateway_api_key=None,
            qmt_gateway_hmac_secret=None,
            deepseek_api_key=None,
        )
    )

    registry = orchestrator._build_registry()
    registered_names = {item["name"] for item in registry.list_tools()}
    llm_tools = registry.to_legacy_registry().deepseek_tools_for_llm()
    llm_names = {item.name for item in llm_tools}

    assert {
        "list_tools",
        "query_bars",
        "plan_remote_data_update",
        "run_remote_data_update",
        "create_factor_spec",
        "generate_factor_code",
        "run_factor_static_checks",
        "evaluate_factor_candidate",
        "detect_tool_gap",
        "generate_tool_tests",
        "run_tool_sandbox_tests",
    }.issubset(registered_names)
    assert {
        "query_bars",
        "plan_remote_data_update",
        "run_remote_data_update",
        "create_factor_spec",
        "generate_factor_code",
        "detect_tool_gap",
    }.issubset(llm_names)
    assert "propose_tool_registration" in registered_names
    assert "propose_tool_registration" not in llm_names
    assert len(llm_names) > 14
