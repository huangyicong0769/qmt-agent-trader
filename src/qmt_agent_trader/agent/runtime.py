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
        session_id: str | None = None,
        experiment_id: str | None = None,
    ) -> list[DeepSeekTool]:
        return (
            self.agent_registry()
            .to_legacy_registry(
                context_factory=lambda: ToolContext(
                    run_id=run_id,
                    session_id=session_id,
                    experiment_id=experiment_id,
                    requested_by_llm=True,
                    call_mode=ToolCallMode.AUTONOMOUS_AGENT,
                    dry_run=False,
                )
            )
            .deepseek_tools_for_llm()
        )

    def ask(
        self,
        prompt: str,
        *,
        max_rounds: int = 100,
        history: list[dict[str, Any]] | None = None,
        session_id: str | None = None,
        experiment_id: str | None = None,
        system_prompt: str | None = None,
    ) -> DeepSeekToolLoopResult:
        if self.settings.deepseek_api_key is None:
            raise ValueError("DEEPSEEK_API_KEY is required for agent ask")
        client = DeepSeekClient(
            api_key=self.settings.deepseek_api_key.get_secret_value(),
            base_url=self.settings.deepseek_base_url,
            model=self.settings.deepseek_model,
        )
        run_id = new_id("run")
        messages: list[dict[str, Any]] = [
            {
                "role": "system",
                "content": system_prompt or _default_research_system_prompt(),
            }
        ]
        messages.extend(history or [])
        messages.append({"role": "user", "content": prompt})
        return client.run_tool_loop(
            messages=messages,
            tools=self.llm_tools(
                run_id=run_id,
                session_id=session_id,
                experiment_id=experiment_id,
            ),
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


def _default_research_system_prompt() -> str:
    return (
        "You are the QMT research agent. Use tools for local facts and multi-step "
        "research loops. You may read data, write generated research artifacts, "
        "generate candidate code in the sandbox, and run simulated backtests. You "
        "must not submit live orders, modify live config, or bypass approvals. "
        "External MCP tools may appear with the configured prefix and follow the "
        "same permission, audit, and evidence rules as native tools. For local "
        "quant research, prefer native data, factor, backtest, and report tools; "
        "do not call external MCP/web tools unless the user explicitly asks for "
        "external news/web context or native tools cannot answer the request. For factor "
        "loops, use list_saved_factors, list_data_catalog, query_universe/query_bars, "
        "query_fundamentals_pit/query_macro_series_pit when relevant, "
        "create_factor_spec, generate_factor_code, run_factor_static_checks, "
        "save_factor, evaluate_factor_candidate, run_backtest, and "
        "generate_research_report. generate_factor_code accepts python_function as "
        "the primary path for unrestricted agent-authored research factors; if a "
        "formula fallback returns NEEDS_PYTHON_FUNCTION, retry with python_function "
        "instead of narrowing the research question. For any factor or strategy research, "
        "first define or resolve an explicit universe. Use create_universe_spec, "
        "validate_universe_spec, build_universe, save_universe_spec, list_universes, "
        "inspect_universe, and query_universe; do not hand-write symbols unless the user "
        "explicitly supplied them or a universe tool returned validated symbols. Generated "
        "universes are research-only and not live-trading-approved. For rolling universes, "
        "disclose per-date symbol resolution and empty-date diagnostics. If "
        "query_fundamentals_pit returns NO_DATA/PARTIAL_COVERAGE "
        "or INVALID_REQUEST, call list_tushare_capabilities, then plan_tushare_fetch "
        "with exact endpoint, ts_code, and field names, then when live update is allowed "
        "call run_tushare_fetch with execute_plan=true and verify with "
        "query_fundamentals_pit. If query_macro_series_pit returns NO_DATA or "
        "INVALID_REQUEST, use list_tushare_capabilities and plan_tushare_fetch for the "
        "exact macro endpoint; when live update is allowed execute with "
        "run_tushare_fetch execute_plan=true and verify with query_macro_series_pit. "
        "If generate_factor_code "
        "returns STATIC_CHECK_FAILED or run_factor_static_checks returns semantic "
        "mismatch, do not save, evaluate, rank, or recommend that generated factor. "
        "For strategy loops, prefer create_strategy_spec, generate_strategy_code, "
        "run_strategy_static_checks, save_strategy_candidate or save_strategy_spec_draft, "
        "run_backtest, and generate_research_report. Keep the final answer at the "
        "same abstraction level as the user's question: if the user asks for strategies, "
        "factor tests are intermediate evidence and must not be presented as strategy "
        "results. Do not claim that code was generated, static checks passed, candidates "
        "were saved, data was refreshed, or a specific universe was used unless a tool "
        "result or tool audit explicitly shows it. Treat adapter_limitations, warnings, "
        "diagnostics "
        "FAIL/BLOCKED/NOT_COMPUTED, missing data, and REVIEW_REQUIRED as material "
        "limitations. If a result used default_universe, unresolved universe, stale "
        "data, or review_required=true, disclose it. The system preserves your raw "
        "final answer and separately reports evidence-output conflicts; do not hide "
        "or smooth over tool statuses to make the answer look cleaner. Do not claim a token or "
        "external API is unavailable unless a tool returned NOT_CONFIGURED, "
        "ENV_BLOCKED, or an explicit upstream/API error; if a tool returns BLOCKED "
        "for missing ts_code or unsupported basket live fill, call it a tool scope "
        "or adapter limitation. Do not blame replay, validation, or test protocols for data "
        "or tool limitations; attribute blockers to the observed tool status, "
        "missing inputs, adapter capability, external API state, or data coverage. "
        "When asked what happened in a previous/current run or what "
        "difficulties were encountered, use get_experiment_tool_calls or "
        "search_experiments when available for the current session first, and mention "
        "only observed tool evidence. Always respond in Chinese."
    )


def _optional_broker_client(settings: Settings) -> RemoteQMTBrokerClient | None:
    if settings.qmt_gateway_api_key is None or settings.qmt_gateway_hmac_secret is None:
        return None
    return RemoteQMTBrokerClient(
        base_url=settings.qmt_gateway_base_url,
        api_key=settings.qmt_gateway_api_key.get_secret_value(),
        hmac_secret=settings.qmt_gateway_hmac_secret.get_secret_value(),
    )
