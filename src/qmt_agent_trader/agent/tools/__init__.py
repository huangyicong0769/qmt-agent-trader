"""Agent tools package — provides tool wiring and registry assembly."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from qmt_agent_trader.agent.tool_registry import AgentToolRegistry

from qmt_agent_trader.agent.audit import AuditLogger
from qmt_agent_trader.agent.experiment_store import ExperimentStore
from qmt_agent_trader.agent.permissions import PermissionLevel
from qmt_agent_trader.agent.sandbox import CodeSandbox
from qmt_agent_trader.agent.schemas import ToolContext, ToolSpec
from qmt_agent_trader.agent.tools.base import tool
from qmt_agent_trader.agent.tools.basic_tools import (
    get_current_time_tool,
    run_shell_command_tool,
)
from qmt_agent_trader.agent.tools.experiment_tools import (
    log_experiment_event_tool,
    search_experiments_tool,
)
from qmt_agent_trader.agent.tools.factor_tools import (
    create_factor_spec_tool,
    evaluate_factor_candidate_tool,
    generate_factor_code_tool,
    list_saved_factors_tool,
    run_factor_static_checks_tool,
    save_factor_tool,
)
from qmt_agent_trader.agent.tools.meta_tools import (
    create_tool_spec_tool,
    detect_tool_gap_tool,
    generate_tool_code_tool,
    generate_tool_tests_tool,
    propose_tool_registration_tool,
    run_tool_sandbox_tests_tool,
    score_tool_candidate_tool,
)
from qmt_agent_trader.agent.tools.query_tools import (
    list_data_catalog_tool,
    query_bars_tool,
    query_universe_tool,
)
from qmt_agent_trader.agent.tools.remote_data_tools import (
    run_remote_data_update_tool,
)
from qmt_agent_trader.agent.tools.strategy_tools import (
    create_strategy_spec_tool,
    generate_research_report_tool,
    generate_strategy_code_tool,
    list_strategy_candidates_tool,
    run_backtest_tool,
)
from qmt_agent_trader.core.config import Settings, get_settings
from qmt_agent_trader.data.storage import DataLake

# Avoid circular import: import AgentToolRegistry inside build_agent_registry


def build_agent_registry(
    *,
    data_lake: DataLake,
    audit_path: Path,
    experiment_root: Path,
    settings: Settings | None = None,
    sandbox: CodeSandbox | None = None,
) -> AgentToolRegistry:
    """Assemble the full AgentToolRegistry with all 16+ MVP tools wired."""
    from qmt_agent_trader.agent.tool_registry import AgentToolRegistry as _ATR
    from qmt_agent_trader.agent.tools import (
        basic_tools,
        experiment_tools,
        factor_tools,
        meta_tools,
        query_tools,
        remote_data_tools,
        strategy_tools,
    )

    resolved_settings = settings or get_settings()
    sb = sandbox or CodeSandbox()
    store = ExperimentStore(experiment_root)
    audit = AuditLogger(audit_path)

    # Wire singletons
    experiment_tools.set_experiment_store(store)
    query_tools.set_data_lake(data_lake)
    basic_tools.wire(settings=resolved_settings)
    remote_data_tools.wire(data_lake=data_lake, settings=resolved_settings)
    factor_tools.wire(sb, store, data_lake)
    strategy_tools.wire(sb, store, data_lake)
    meta_tools.wire(sb, store)

    registry = _ATR(audit_logger=audit)

    # Build list_tools / describe_tools inline (they need the registry ref)
    def _list_tools(input_data: dict[str, Any], context: ToolContext) -> dict[str, Any]:
        permission = input_data.get("permission_level")
        return {"tools": registry.list_tools(permission=permission)}

    def _describe_tool(input_data: dict[str, Any], context: ToolContext) -> dict[str, Any]:
        name = input_data.get("name", "")
        spec = registry.describe_tool(name)
        return {"tool_spec": spec.model_dump(mode="json")}

    registry.register(
        tool(
            ToolSpec(
                name="list_tools",
                description="列出当前 Agent 可用工具，可按权限过滤。",
                permission=PermissionLevel.READ_ONLY,
                input_schema={
                    "type": "object",
                    "properties": {
                        "permission_level": {
                            "type": "string",
                            "description": "Optional permission filter.",
                        }
                    },
                },
                output_schema={"type": "object", "properties": {"tools": {"type": "array"}}},
                deterministic=False,
            ),
            fn=_list_tools,
        )
    )
    registry.register(
        tool(
            ToolSpec(
                name="describe_tool",
                description="查看指定工具的详细 schema 和限制。",
                permission=PermissionLevel.READ_ONLY,
                input_schema={
                    "type": "object",
                    "properties": {"name": {"type": "string"}},
                    "required": ["name"],
                },
                output_schema={"type": "object", "properties": {"tool_spec": {"type": "object"}}},
                deterministic=True,
            ),
            fn=_describe_tool,
        )
    )

    registry.register_all(
        # Experiment tools
        log_experiment_event_tool,
        search_experiments_tool,
        # Data/query tools
        list_data_catalog_tool,
        query_universe_tool,
        query_bars_tool,
        run_remote_data_update_tool,
        run_shell_command_tool,
        get_current_time_tool,
        # Factor tools
        list_saved_factors_tool,
        create_factor_spec_tool,
        generate_factor_code_tool,
        run_factor_static_checks_tool,
        save_factor_tool,
        evaluate_factor_candidate_tool,
        # Strategy tools
        create_strategy_spec_tool,
        generate_strategy_code_tool,
        list_strategy_candidates_tool,
        run_backtest_tool,
        generate_research_report_tool,
        # Meta tools
        detect_tool_gap_tool,
        create_tool_spec_tool,
        generate_tool_code_tool,
        generate_tool_tests_tool,
        run_tool_sandbox_tests_tool,
        score_tool_candidate_tool,
        propose_tool_registration_tool,
    )
    return registry
