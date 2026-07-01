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
        "query_macro_series_pit",
        "run_remote_data_update",
        "run_fundamental_data_update",
        "run_macro_data_update",
        "run_shell_command",
        "create_factor_spec",
        "generate_factor_code",
        "list_saved_factors",
        "run_factor_static_checks",
        "save_factor",
        "evaluate_factor_candidate",
        "list_strategy_candidates",
        "get_experiment_tool_calls",
        "save_strategy_candidate",
        "detect_tool_gap",
        "generate_tool_tests",
        "run_tool_sandbox_tests",
    }.issubset(registered_names)
    assert "plan_remote_data_update" not in registered_names
    assert "query_fundamentals_pit" in registered_names
    assert "run_shell_command" not in llm_names
    assert {
        "query_bars",
        "query_macro_series_pit",
        "run_remote_data_update",
        "run_fundamental_data_update",
        "run_macro_data_update",
        "create_factor_spec",
        "generate_factor_code",
        "list_saved_factors",
        "save_factor",
        "list_strategy_candidates",
        "get_experiment_tool_calls",
        "save_strategy_candidate",
        "detect_tool_gap",
    }.issubset(llm_names)
    assert "plan_remote_data_update" not in llm_names
    assert "query_fundamentals_pit" in llm_names
    assert "propose_tool_registration" in registered_names
    assert "propose_tool_registration" not in llm_names
    assert len(llm_names) > 14
