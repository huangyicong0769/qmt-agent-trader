"""Agent Orchestrator — bridges Router → LLM Runtime → EventBus → Frontend.

Takes a natural language message + routing decision, executes the LLM
tool loop with real-time event streaming suitable for SSE delivery.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from qmt_agent_trader.agent.llm_client import (
    DeepSeekClient,
)
from qmt_agent_trader.agent.permissions import ToolCapability
from qmt_agent_trader.agent.router import RoutingDecision
from qmt_agent_trader.agent.sandbox import CodeSandbox
from qmt_agent_trader.agent.schemas import ToolContext
from qmt_agent_trader.agent.tool_registry import AgentToolRegistry, ToolDefinition, ToolRegistry
from qmt_agent_trader.agent.tools import build_agent_registry
from qmt_agent_trader.agent.tools.backtest_tools import (
    plan_sensitivity_analysis,
    run_factor_rank_sensitivity,
    run_factor_rank_sensitivity_report,
)
from qmt_agent_trader.agent.tools.research_context import get_research_context
from qmt_agent_trader.backtest.service import (
    compare_backtest_reports,
    run_backtest_report,
)
from qmt_agent_trader.core.config import Settings, get_settings
from qmt_agent_trader.core.ids import new_id
from qmt_agent_trader.data.bars import load_daily_bars
from qmt_agent_trader.data.storage import DataLake
from qmt_agent_trader.factors.service import (
    compute_factor_to_lake,
    validate_factor,
    walk_forward_factor_validation,
)
from qmt_agent_trader.services.research_report_service import compare_research_reports
from qmt_agent_trader.strategy.approval import read_approval_file

# ── Event wrapper for SSE streaming ──


@dataclass
class OrchestratorEvent:
    """Lightweight event emitted during orchestration, suitable for SSE JSON."""

    type: str  # e.g. "progress", "tool_start", "tool_done", "llm_message", "done", "error"
    run_id: str
    data: dict[str, Any] = field(default_factory=dict)
    message: str = ""

    def to_sse(self) -> str:
        payload = {
            "type": self.type,
            "run_id": self.run_id,
            "message": self.message,
            "data": self.data,
        }
        return json.dumps(payload, ensure_ascii=False, default=str)


# ── Orchestrator ──


class AgentOrchestrator:
    """Execute LLM tool loops with real-time event streaming."""

    def __init__(
        self,
        settings: Settings | None = None,
        data_lake: DataLake | None = None,
    ) -> None:
        resolved = settings or get_settings()
        self.settings = resolved
        self._lake = data_lake or DataLake(
            root=resolved.resolved_data_dir / "lake",
            duckdb_path=resolved.resolved_data_dir / "qmt_agent_trader.duckdb",
        )
        self._reports_dir = resolved.project_root / "reports" / "backtests"
        self._research_dir = resolved.project_root / "reports" / "research"
        self._approvals_dir = resolved.project_root / "approvals"

    @property
    def lake(self) -> DataLake:
        return self._lake

    def _build_registry(self) -> AgentToolRegistry:
        """Build the full AgentTool registry used by chat orchestration."""
        return build_agent_registry(
            data_lake=self._lake,
            audit_path=self.settings.resolved_log_dir / "audit" / "agent_tool_calls.jsonl",
            experiment_root=self.settings.resolved_data_dir / "experiments",
            settings=self.settings,
            sandbox=CodeSandbox(),
        )

    async def execute_stream(
        self,
        message: str,
        routing: RoutingDecision | None = None,
        *,
        run_id: str | None = None,
        max_rounds: int = 100,
        cancel_event: asyncio.Event | None = None,
    ) -> AsyncGenerator[OrchestratorEvent, None]:
        """Execute the LLM tool loop and yield events suitable for SSE streaming.

        If cancel_event is set, the generator checks it during the polling loop
        and yields a 'cancelled' event instead of completing normally.
        """
        rid = run_id or new_id("run")
        experiment_id = new_id("exp")

        # ── Emit start ──
        yield OrchestratorEvent(
            type="run_started",
            run_id=rid,
            data={
                "experiment_id": experiment_id,
                "intent": routing.intent.value if routing else "GENERAL_RESEARCH",
                "confidence": routing.confidence if routing else 0.0,
                "rationale": routing.rationale if routing else "",
            },
            message=f"Starting run {rid}",
        )

        # ── Check LLM availability ──
        if self.settings.deepseek_api_key is None:
            yield OrchestratorEvent(
                type="error",
                run_id=rid,
                message="DeepSeek API key not configured. Set DEEPSEEK_API_KEY in .env.",
                data={"error": "llm_not_configured"},
            )
            return

        try:
            client = DeepSeekClient(
                api_key=self.settings.deepseek_api_key.get_secret_value(),
                base_url=self.settings.deepseek_base_url,
                model=self.settings.deepseek_model,
            )
        except Exception as exc:
            yield OrchestratorEvent(
                type="error",
                run_id=rid,
                message=f"Failed to initialize LLM: {exc}",
                data={"error": str(exc)},
            )
            return

        # ── Build system prompt ──
        routing_hint = ""
        if routing and routing.proposed_workflow:
            routing_hint = (
                f"\nThe user's intent has been classified as: {routing.intent.value}. "
                f"Suggested workflow: {routing.proposed_workflow}. "
                f"Recommended tools: {', '.join(routing.required_tools[:6])}. "
                f"Rationale: {routing.rationale}"
            )

        system_msg = (
            "You are the QMT research agent. Use tools for local facts and for "
            "multi-step research loops. You may read data, write generated research "
            "artifacts, generate candidate code in the sandbox, and run simulated "
            "backtests. You must not submit live orders, modify live config, or bypass "
            "approvals. Tools that require human approval are not available to you as "
            "function calls. Use list_tools and describe_tool when you need to discover "
            "the available surface. Keep calling distinct useful tools until you have "
            "enough evidence to answer; avoid repeating the same tool with the same "
            "arguments. If a tool returns NOT_AVAILABLE or an error, either choose a "
            "different relevant tool or report the blocker clearly. Always respond in "
            "Chinese. Typical factor loop: list_data_catalog, query_universe/query_bars, "
            "create_factor_spec, generate_factor_code, run_factor_static_checks, "
            "evaluate_factor_candidate, generate_research_report. Typical strategy "
            "loop: search_experiments, create_strategy_spec, generate_strategy_code, "
            "run_backtest, generate_research_report. Typical self-bootstrap loop: "
            "search_experiments, detect_tool_gap, create_tool_spec, generate_tool_code, "
            "generate_tool_tests, run_tool_sandbox_tests, score_tool_candidate."
            + routing_hint
        )

        registry = self._build_registry()
        legacy_registry = registry.to_legacy_registry(
            context_factory=lambda: ToolContext(
                run_id=rid,
                experiment_id=experiment_id,
                requested_by_llm=True,
                dry_run=True,
            )
        )
        tools = legacy_registry.deepseek_tools_for_llm()

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": message},
        ]

        yield OrchestratorEvent(
            type="progress",
            run_id=rid,
            message=f"LLM initialised with {len(tools)} tools. Starting tool loop...",
            data={"tool_count": len(tools), "model": self.settings.deepseek_model},
        )

        # ── Run the tool loop in a thread to avoid blocking the event loop ──
        import asyncio as _asyncio
        import concurrent.futures as _futures

        stream_buffer: list[Any] = []
        stream_error_msg: str | None = None

        def _run_stream() -> None:
            nonlocal stream_error_msg
            try:
                for evt in client.run_tool_loop_stream(
                    messages=messages, tools=tools, max_rounds=max_rounds
                ):
                    stream_buffer.append(evt)
            except Exception as exc:
                stream_error_msg = str(exc)

        with _futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(_run_stream)
            yielded = 0

            while not future.done() or yielded < len(stream_buffer):
                while yielded < len(stream_buffer):
                    evt = stream_buffer[yielded]
                    yielded += 1
                    for oe in _stream_to_events(evt, rid, experiment_id):
                        yield oe
                if future.done():
                    break
                # Check for cancellation
                if cancel_event is not None and cancel_event.is_set():
                    stream_error_msg = "cancelled"
                    future.cancel()
                    break
                await _asyncio.sleep(0.05)

            if stream_error_msg == "cancelled":
                yield OrchestratorEvent(
                    type="cancelled",
                    run_id=rid,
                    message="Execution cancelled by user.",
                    data={"reason": "user_interrupt"},
                )
                return

            if stream_error_msg:
                yield OrchestratorEvent(
                    type="error",
                    run_id=rid,
                    message=f"LLM streaming error: {stream_error_msg}",
                    data={"error": stream_error_msg},
                )
                return

        # ── Count tool calls ──
        tcount = sum(
            1 for e in stream_buffer if e.__class__.__name__ == "ToolResult"
        )
        yield OrchestratorEvent(
            type="done",
            run_id=rid,
            message="Run completed successfully.",
            data={
                "experiment_id": experiment_id,
                "tool_calls_count": tcount,
                "intent": routing.intent.value if routing else "GENERAL_RESEARCH",
            },
        )


# ── Stream event converter ──


def _stream_to_events(
    evt: Any, run_id: str, experiment_id: str
) -> list[OrchestratorEvent]:
    """Convert a StreamEvent to one or more OrchestratorEvents."""
    cls_name = evt.__class__.__name__

    if cls_name == "TextDelta":
        return [
            OrchestratorEvent(
                type="token",
                run_id=run_id,
                message=evt.content,
                data={"token": evt.content, "experiment_id": experiment_id},
            )
        ]

    if cls_name == "ToolCallStart":
        return [
            OrchestratorEvent(
                type="tool_start",
                run_id=run_id,
                message=f"Calling: {evt.tool_name}",
                data={
                    "tool_name": evt.tool_name,
                    "tool_call_id": evt.tool_call_id,
                    "experiment_id": experiment_id,
                },
            )
        ]

    if cls_name == "ToolCallComplete":
        return [
            OrchestratorEvent(
                type="tool_args",
                run_id=run_id,
                message=f"Args ready: {evt.tool_name}",
                data={
                    "tool_name": evt.tool_name,
                    "arguments": evt.arguments,
                    "experiment_id": experiment_id,
                },
            )
        ]

    if cls_name == "ToolResult":
        preview = _preview(evt.result)
        return [
            OrchestratorEvent(
                type="tool_done",
                run_id=run_id,
                message=f"Tool: {evt.tool_name} ✓",
                data={
                    "tool_name": evt.tool_name,
                    "result_preview": preview,
                    "experiment_id": experiment_id,
                },
            )
        ]

    if cls_name == "LoopError":
        return [
            OrchestratorEvent(
                type="error",
                run_id=run_id,
                message=evt.message,
                data={"error": evt.message},
            )
        ]

    if cls_name == "LoopBreak":
        return [
            OrchestratorEvent(
                type="progress",
                run_id=run_id,
                message=f"⛔ {evt.message}",
                data={"warning": evt.message},
            )
        ]

    if cls_name == "SafetyCapHit":
        return [
            OrchestratorEvent(
                type="progress",
                run_id=run_id,
                message=f"🛑 {evt.message}",
                data={"warning": evt.message},
            )
        ]

    return []


# ── Tool registry builder (forked from AgentRuntime for v1 compatibility) ──


def _register_all_tools(
    registry: ToolRegistry,
    lake: DataLake,
    reports_dir: Path,
    research_dir: Path,
    approvals_dir: Path,
) -> None:
    registry.register(
        ToolDefinition(
            name="get_research_context",
            capability=ToolCapability.READ_DATA,
            description="Return local research capabilities, constraints, and LLM boundaries.",
            parameters=_object_schema(
                {"universe": {"type": "string", "description": "Comma-separated universe names."}},
                required=["universe"],
            ),
            fn=get_research_context,
        )
    )
    registry.register(
        ToolDefinition(
            name="list_datasets",
            capability=ToolCapability.READ_DATA,
            description="List datasets in the local DuckDB/Parquet data lake.",
            parameters=_object_schema(
                {
                    "layer": {
                        "type": "string",
                        "description": "Optional layer: raw, bronze, silver, or gold.",
                    },
                    "prefix": {"type": "string", "description": "Optional dataset prefix."},
                }
            ),
            fn=lambda layer=None, prefix=None: _list_datasets(lake, layer=layer, prefix=prefix),
        )
    )
    registry.register(
        ToolDefinition(
            name="summarize_daily_bars",
            capability=ToolCapability.READ_DATA,
            description="Summarize canonical daily bars and trade-state counts.",
            parameters=_object_schema(
                {
                    "start": {"type": "string", "description": "Optional start date."},
                    "end": {"type": "string", "description": "Optional end date."},
                }
            ),
            fn=lambda start=None, end=None: _summarize_bars(lake, start=start, end=end),
        )
    )
    registry.register(
        ToolDefinition(
            name="list_factors",
            capability=ToolCapability.READ_DATA,
            description="List built-in daily factor names.",
            parameters=_object_schema({}),
            fn=_list_factors,
        )
    )
    registry.register(
        ToolDefinition(
            name="compute_factor",
            capability=ToolCapability.WRITE_RESEARCH,
            description="Compute a built-in factor for one date and write it to the gold layer.",
            parameters=_object_schema(
                {
                    "name": {"type": "string", "description": "Factor name."},
                    "date": {"type": "string", "description": "Target date."},
                },
                required=["name", "date"],
            ),
            fn=lambda name, date: compute_factor_to_lake(
                lake, name=name, date=date
            ).as_dict(),
        )
    )
    registry.register(
        ToolDefinition(
            name="validate_factor",
            capability=ToolCapability.RUN_BACKTEST,
            description="Validate a built-in factor over a date range.",
            parameters=_object_schema(
                {
                    "name": {"type": "string", "description": "Factor name."},
                    "start": {"type": "string", "description": "Start date."},
                    "end": {"type": "string", "description": "End date."},
                },
                required=["name", "start", "end"],
            ),
            fn=lambda name, start, end: validate_factor(
                lake, name=name, start=start, end=end
            ).as_dict(),
        )
    )
    registry.register(
        ToolDefinition(
            name="walk_forward_factor_validation",
            capability=ToolCapability.RUN_BACKTEST,
            description=(
                "Validate a built-in factor across rolling walk-forward windows "
                "using daily IC and long-short spread. Research-only."
            ),
            parameters=_object_schema(
                {
                    "name": {"type": "string", "description": "Factor name."},
                    "start": {"type": "string", "description": "Start date."},
                    "end": {"type": "string", "description": "End date."},
                    "window_days": {"type": "integer", "description": "Trading-day window."},
                    "step_days": {"type": "integer", "description": "Trading-day step."},
                    "quantile": {"type": "number", "description": "Top/bottom quantile."},
                },
                required=["name", "start", "end"],
            ),
            fn=lambda name, start, end, window_days=63, step_days=63, quantile=0.20: (
                walk_forward_factor_validation(
                    lake, name=name, start=start, end=end,
                    window_days=window_days, step_days=step_days, quantile=quantile,
                ).as_dict()
            ),
        )
    )
    registry.register(
        ToolDefinition(
            name="run_backtest",
            capability=ToolCapability.RUN_BACKTEST,
            description="Run a daily T+1 simulated backtest and persist a report.",
            parameters=_object_schema(
                {
                    "symbol": {"type": "string"},
                    "signal_date": {"type": "string"},
                    "quantity": {"type": "integer"},
                }
            ),
            fn=lambda symbol=None, signal_date=None, quantity=100: run_backtest_report(
                lake, reports_dir=reports_dir, symbol=symbol,
                signal_date=signal_date, quantity=quantity,
            ).as_dict(),
        )
    )
    registry.register(
        ToolDefinition(
            name="compare_backtests",
            capability=ToolCapability.RUN_BACKTEST,
            description="Compare recent persisted backtest reports.",
            parameters=_object_schema(
                {"limit": {"type": "integer"}}
            ),
            fn=lambda limit=10: compare_backtest_reports(reports_dir, limit=limit),
        )
    )
    registry.register(
        ToolDefinition(
            name="plan_sensitivity_analysis",
            capability=ToolCapability.RUN_BACKTEST,
            description="Build a robustness scenario matrix. Research-only.",
            parameters=_object_schema(
                {
                    "cost_multipliers": {"type": "array", "items": {"type": "number"}},
                    "slippage_bps": {"type": "array", "items": {"type": "number"}},
                    "execution_delay_days": {"type": "array", "items": {"type": "integer"}},
                    "top_n": {"type": "array", "items": {"type": "integer"}},
                    "max_single_position_pct": {"type": "array", "items": {"type": "number"}},
                }
            ),
            fn=plan_sensitivity_analysis,
        )
    )
    registry.register(
        ToolDefinition(
            name="run_factor_rank_sensitivity",
            capability=ToolCapability.RUN_BACKTEST,
            description="Run factor-rank robustness simulation. Research-only.",
            parameters=_object_schema(
                {
                    "factor_name": {"type": "string"},
                    "cost_multipliers": {"type": "array", "items": {"type": "number"}},
                    "slippage_bps": {"type": "array", "items": {"type": "number"}},
                    "execution_delay_days": {"type": "array", "items": {"type": "integer"}},
                    "top_n": {"type": "array", "items": {"type": "integer"}},
                    "max_single_position_pct": {"type": "array", "items": {"type": "number"}},
                    "initial_cash": {"type": "number"},
                },
                required=["factor_name"],
            ),
            fn=lambda factor_name, **kw: run_factor_rank_sensitivity(
                lake, factor_name=factor_name,
                cost_multipliers=kw.get("cost_multipliers"),
                slippage_bps=kw.get("slippage_bps"),
                execution_delay_days=kw.get("execution_delay_days"),
                top_n=kw.get("top_n"),
                max_single_position_pct=kw.get("max_single_position_pct"),
                initial_cash=kw.get("initial_cash", 1_000_000.0),
            ),
        )
    )
    registry.register(
        ToolDefinition(
            name="run_factor_rank_sensitivity_report",
            capability=ToolCapability.WRITE_RESEARCH,
            description="Run factor-rank analysis and persist research evidence.",
            parameters=_object_schema(
                {
                    "factor_name": {"type": "string"},
                    "cost_multipliers": {"type": "array", "items": {"type": "number"}},
                    "slippage_bps": {"type": "array", "items": {"type": "number"}},
                    "execution_delay_days": {"type": "array", "items": {"type": "integer"}},
                    "top_n": {"type": "array", "items": {"type": "integer"}},
                    "max_single_position_pct": {"type": "array", "items": {"type": "number"}},
                    "initial_cash": {"type": "number"},
                    "agent_notes": {"type": "string"},
                },
                required=["factor_name"],
            ),
            fn=lambda factor_name, **kw: run_factor_rank_sensitivity_report(
                lake, research_dir, factor_name=factor_name,
                cost_multipliers=kw.get("cost_multipliers"),
                slippage_bps=kw.get("slippage_bps"),
                execution_delay_days=kw.get("execution_delay_days"),
                top_n=kw.get("top_n"),
                max_single_position_pct=kw.get("max_single_position_pct"),
                initial_cash=kw.get("initial_cash", 1_000_000.0),
                agent_notes=kw.get("agent_notes"),
            ),
        )
    )
    registry.register(
        ToolDefinition(
            name="compare_research_reports",
            capability=ToolCapability.READ_DATA,
            description="Compare recent persisted research evidence packages.",
            parameters=_object_schema(
                {"limit": {"type": "integer"}}
            ),
            fn=lambda limit=10: compare_research_reports(research_dir, limit=limit),
        )
    )
    registry.register(
        ToolDefinition(
            name="list_strategy_approvals",
            capability=ToolCapability.READ_DATA,
            description="List local strategy approval files and paper/live flags.",
            parameters=_object_schema({}),
            fn=lambda: _list_approvals(approvals_dir),
        )
    )


# ── Helpers ──


def _list_datasets(
    lake: DataLake, *, layer: str | None = None, prefix: str | None = None
) -> dict[str, object]:
    layers = [layer] if layer else ["raw", "bronze", "silver", "gold"]
    return {
        "layers": {
            item: lake.list_dataset_names(item, prefix=prefix) for item in layers
        }
    }


def _summarize_bars(
    lake: DataLake, *, start: str | None = None, end: str | None = None
) -> dict[str, object]:
    bars = load_daily_bars(lake, start=start, end=end)
    if bars.empty:
        return {"status": "empty", "rows": 0}
    return {
        "status": "ok",
        "rows": len(bars),
        "symbols": int(bars["symbol"].nunique()),
        "start": f"{pd.to_datetime(bars['trade_date'].min()).date():%Y%m%d}",
        "end": f"{pd.to_datetime(bars['trade_date'].max()).date():%Y%m%d}",
        "trade_state_counts": {
            "suspended": int(bars["suspended"].sum()),
            "limit_up": int(bars["limit_up"].sum()),
            "limit_down": int(bars["limit_down"].sum()),
            "st": int(bars["st"].sum()),
        },
    }


def _list_factors() -> dict[str, object]:
    return {
        "factors": [
            "momentum_20d",
            "momentum_60d",
            "reversal_5d",
            "volatility_20d",
            "turnover_20d",
            "amount_zscore_20d",
        ]
    }


def _list_approvals(directory: Path) -> dict[str, object]:
    if not directory.exists():
        return {"approvals": []}
    approvals = []
    for path in sorted(directory.glob("*.approval.yaml")):
        approval = read_approval_file(path)
        approvals.append({
            "strategy_id": approval.strategy_id,
            "strategy_version": approval.strategy_version,
            "paper_trading_allowed": approval.paper_trading_allowed,
            "live_trading_allowed": approval.live_trading_allowed,
            "path": str(path),
        })
    return {"approvals": approvals}


def _object_schema(
    properties: dict[str, dict[str, Any]], *, required: list[str] | None = None
) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": properties,
        "required": required or [],
        "additionalProperties": False,
    }


def _safe_result(result: Any) -> Any:
    """Make a tool result JSON-safe."""
    try:
        json.dumps(result, default=str)
        return result
    except (TypeError, ValueError):
        return str(result)


def _preview(result: Any, max_len: int = 200) -> str:
    """Create a short preview of a tool result."""
    s = _safe_result(result)
    if isinstance(s, str):
        return s[:max_len] + ("..." if len(s) > max_len else "")
    try:
        j = json.dumps(s, ensure_ascii=False, default=str)
        return j[:max_len] + ("..." if len(j) > max_len else "")
    except Exception:
        return str(s)[:max_len] + "..."
