"""Unified runtime facade for Agent tool execution."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from qmt_agent_trader.agent.llm_client import DeepSeekClient, DeepSeekTool, DeepSeekToolLoopResult
from qmt_agent_trader.agent.permissions import ToolCallMode
from qmt_agent_trader.agent.schemas import ToolContext, ToolSpec
from qmt_agent_trader.broker.remote_client import RemoteQMTBrokerClient
from qmt_agent_trader.core.config import Settings, get_settings
from qmt_agent_trader.core.ids import new_id
from qmt_agent_trader.data.storage import DataLake

if TYPE_CHECKING:
    from qmt_agent_trader.agent.tool_registry import AgentToolRegistry


@dataclass
class AgentRuntime:
    """Single facade shared by Web, CLI, workflows, and LLM tool loops."""

    settings: Settings
    lake: DataLake
    reports_dir: Path
    research_reports_dir: Path
    approvals_dir: Path
    broker_client: RemoteQMTBrokerClient | None = None
    _agent_registry: AgentToolRegistry | None = field(default=None, init=False, repr=False)

    def agent_registry(self) -> AgentToolRegistry:
        from qmt_agent_trader.agent.sandbox import CodeSandbox
        from qmt_agent_trader.agent.tools import build_agent_registry

        if self._agent_registry is None:
            self._agent_registry = build_agent_registry(
                data_lake=self.lake,
                audit_path=self.settings.resolved_log_dir / "audit" / "agent_tool_calls.jsonl",
                experiment_root=self.settings.resolved_data_dir / "experiments",
                settings=self.settings,
                sandbox=CodeSandbox(),
            )
        return self._agent_registry

    def list_tools(
        self,
        *,
        permission: str | None = None,
        agent_callable_only: bool = True,
        call_mode: ToolCallMode = ToolCallMode.AUTONOMOUS_AGENT,
    ) -> list[dict[str, object]]:
        return self.agent_registry().list_tools(
            permission=permission,
            agent_callable_only=agent_callable_only,
            call_mode=call_mode,
        )

    def describe_tool(self, name: str) -> ToolSpec:
        return self.agent_registry().describe_tool(name)

    def run_tool(
        self,
        name: str,
        input_data: dict[str, Any],
        context: ToolContext | None = None,
    ) -> dict[str, Any]:
        return self.agent_registry().run_tool(
            name,
            input_data,
            context
            or ToolContext(
                run_id="runtime",
                requested_by_llm=True,
                call_mode=ToolCallMode.AUTONOMOUS_AGENT,
                dry_run=False,
            ),
        )

    def llm_tools(
        self,
        *,
        run_id: str,
        experiment_id: str | None = None,
    ) -> list[DeepSeekTool]:
        return (
            self.agent_registry()
            .to_legacy_registry(
                context_factory=lambda: ToolContext(
                    run_id=run_id,
                    experiment_id=experiment_id,
                    requested_by_llm=True,
                    call_mode=ToolCallMode.AUTONOMOUS_AGENT,
                    dry_run=False,
                )
            )
            .deepseek_tools_for_llm()
        )

    def ask(self, prompt: str, *, max_rounds: int = 100) -> DeepSeekToolLoopResult:
        if self.settings.deepseek_api_key is None:
            raise ValueError("DEEPSEEK_API_KEY is required for agent ask")
        client = DeepSeekClient(
            api_key=self.settings.deepseek_api_key.get_secret_value(),
            base_url=self.settings.deepseek_base_url,
            model=self.settings.deepseek_model,
        )
        run_id = new_id("run")
        return client.run_tool_loop(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are the QMT research agent. Use tools for local facts and "
                        "multi-step research loops. You may read data, write generated "
                        "research artifacts, generate candidate code in the sandbox, and run "
                        "simulated backtests. You must not submit live orders, modify live "
                        "config, or bypass approvals. External MCP tools may appear with "
                        "the configured prefix and follow the same permission, audit, and "
                        "evidence rules as native tools. Always respond in Chinese."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            tools=self.llm_tools(run_id=run_id),
            max_rounds=max_rounds,
        )


def build_default_runtime(
    settings: Settings | None = None,
    *,
    broker_client: RemoteQMTBrokerClient | None = None,
) -> AgentRuntime:
    resolved = settings or get_settings()
    lake = DataLake(
        root=resolved.resolved_data_dir / "lake",
        duckdb_path=resolved.resolved_data_dir / "qmt_agent_trader.duckdb",
    )
    return AgentRuntime(
        settings=resolved,
        lake=lake,
        reports_dir=resolved.project_root / "reports" / "backtests",
        research_reports_dir=resolved.project_root / "reports" / "research",
        approvals_dir=resolved.project_root / "approvals",
        broker_client=broker_client or _optional_broker_client(resolved),
    )


def _optional_broker_client(settings: Settings) -> RemoteQMTBrokerClient | None:
    if settings.qmt_gateway_api_key is None or settings.qmt_gateway_hmac_secret is None:
        return None
    return RemoteQMTBrokerClient(
        base_url=settings.qmt_gateway_base_url,
        api_key=settings.qmt_gateway_api_key.get_secret_value(),
        hmac_secret=settings.qmt_gateway_hmac_secret.get_secret_value(),
    )
