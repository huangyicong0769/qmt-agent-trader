"""Agent tools package — provides tool wiring and registry assembly."""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from qmt_agent_trader.agent.tool_registry import AgentToolRegistry

from qmt_agent_trader.agent.audit import AuditLogger
from qmt_agent_trader.agent.experiment_store import ExperimentStore
from qmt_agent_trader.agent.permissions import PermissionLevel, ToolCallMode
from qmt_agent_trader.agent.sandbox import CodeSandbox
from qmt_agent_trader.agent.schemas import ToolContext, ToolSpec
from qmt_agent_trader.agent.tool_dependencies import AgentToolDependencies
from qmt_agent_trader.agent.tools.base import tool
from qmt_agent_trader.agent.tools.basic_tools import build_basic_tools
from qmt_agent_trader.agent.tools.experiment_tools import build_experiment_tools
from qmt_agent_trader.agent.tools.factor_tools import build_factor_tools
from qmt_agent_trader.agent.tools.meta_tools import build_meta_tools
from qmt_agent_trader.agent.tools.query_tools import build_query_tools
from qmt_agent_trader.agent.tools.remote_data_tools import build_remote_data_tools
from qmt_agent_trader.agent.tools.strategy_tools import build_strategy_tools
from qmt_agent_trader.agent.tools.todo_tools import build_todo_tools
from qmt_agent_trader.agent.tools.universe_tools import build_universe_tools
from qmt_agent_trader.core.config import Settings, get_settings
from qmt_agent_trader.data.storage import DataLake
from qmt_agent_trader.persistence.atomic_files import AtomicFileStore
from qmt_agent_trader.persistence.cache import ContentAddressedCache
from qmt_agent_trader.persistence.paths import PersistencePaths

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

    resolved_settings = settings or get_settings()
    sb = sandbox or CodeSandbox()
    paths = PersistencePaths.from_settings(resolved_settings)
    atomic_store = AtomicFileStore(data_lake.lock_manager)
    store = ExperimentStore(
        experiment_root.expanduser().resolve(),
        locks_root=paths.locks_root,
        quarantine_root=paths.quarantine_root / "experiments",
    )
    audit = AuditLogger(
        audit_path,
        atomic_store=atomic_store,
        fsync=resolved_settings.audit_fsync,
        rotation_bytes=resolved_settings.audit_rotation_bytes,
    )
    deps = AgentToolDependencies(
        settings=resolved_settings,
        data_lake=data_lake,
        sandbox=sb,
        experiment_store=store,
        audit_logger=audit,
        cache=ContentAddressedCache(
            paths.cache_root,
            atomic_store,
            ttl=timedelta(seconds=resolved_settings.cache_ttl_seconds),
        ),
    )

    registry = _ATR(audit_logger=audit)

    # Build list_tools / describe_tools inline (they need the registry ref)
    def _list_tools(input_data: dict[str, Any], context: ToolContext) -> dict[str, Any]:
        permission = input_data.get("permission_level")
        return {
            "tools": registry.list_tools(
                permission=permission,
                agent_callable_only=True,
                call_mode=context.call_mode or ToolCallMode.AUTONOMOUS_AGENT,
            )
        }

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
        *build_experiment_tools(deps),
        *build_query_tools(deps),
        *build_universe_tools(deps),
        *build_remote_data_tools(deps),
        *build_basic_tools(deps),
        *build_factor_tools(deps),
        *build_strategy_tools(deps),
        *build_meta_tools(deps),
        *build_todo_tools(deps),
    )
    if resolved_settings.mcp_enabled:
        from qmt_agent_trader.agent.mcp_client import build_mcp_tools

        mcp_config_path = resolved_settings.mcp_config_path
        if not mcp_config_path.is_absolute():
            mcp_config_path = resolved_settings.project_root / mcp_config_path
        registry.register_all(
            *build_mcp_tools(
                config_path=mcp_config_path,
                tool_prefix=resolved_settings.mcp_tool_prefix,
                default_timeout_seconds=resolved_settings.mcp_default_timeout_seconds,
            )
        )
    return registry
