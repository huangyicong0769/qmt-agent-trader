"""Agent Orchestrator — bridges LLM Runtime → EventBus → Frontend.

Takes a natural language message and executes the LLM tool loop with
real-time event streaming suitable for SSE delivery.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field
from typing import Any

from qmt_agent_trader.agent.evidence_ledger import EvidenceLedger
from qmt_agent_trader.agent.experiment_store import ExperimentStore
from qmt_agent_trader.agent.llm_client import (
    DeepSeekClient,
)
from qmt_agent_trader.agent.runtime import AgentRuntime, build_default_runtime
from qmt_agent_trader.agent.tool_registry import AgentToolRegistry
from qmt_agent_trader.agent.tool_result import status_icon
from qmt_agent_trader.core.config import Settings, get_settings
from qmt_agent_trader.core.ids import new_id
from qmt_agent_trader.data.storage import DataLake
from qmt_agent_trader.persistence.paths import PersistencePaths

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


SYSTEM_PROMPT_GLOBAL = """
[GLOBAL PRINCIPLES]
- Do not fabricate data, tool calls, code changes, or coverage.
- Tool structured fields are the source of truth: execution_status, domain_status,
  evidence_status, coverage_status, dataset_results, repair_action, and
  verification_action.
- Do not override, hide, or soften PARTIAL, NO_DATA, INVALID_REQUEST, BLOCKED,
  FAILED, or WEAK evidence in the final answer.
""".strip()


SYSTEM_PROMPT_DATA_RULES = """
[DATA TOOL INTERPRETATION RULES]
- execution_status only describes Python/tool execution.
- domain_status describes whether the data-domain action succeeded.
- evidence_status describes whether the result can support research evidence.
- coverage_status describes whether the requested data window is covered.
- For run_tushare_fetch dataset_results, count only status=updated with rows>0
  as successful data updates. Do not summarize success by API call count.
- Any dataset_result with rows=0, status=NO_DATA, PARTIAL_UPDATE,
  SCHEMA_MISMATCH, FAILED, or INVALID_REQUEST is incomplete or invalid evidence.
""".strip()


SYSTEM_PROMPT_REPAIR_RULES = """
[REPAIR & PLANNING RULES]
- If a query tool returns repair_action, execute that concrete action when it
  fits permissions and budget.
- Do not substitute a different endpoint unless list_tushare_capabilities or
  plan_tushare_fetch proves it.
- INVALID_REQUEST means fix arguments first; do not call run_tushare_fetch until
  the request is valid.
""".strip()


SYSTEM_PROMPT_VERIFICATION_RULES = """
[VERIFICATION RULES]
- After run_tushare_fetch, execute verification_action when present.
- Data that was fetched but not verified must not be described as filled.
- For PIT macro data, describe the actual visible window returned by the tool,
  not the requested window.
""".strip()


SYSTEM_PROMPT_FINAL_ANSWER_RULES = """
[FINAL ANSWER CONSTRAINTS]
- Final answers must match dataset_results and coverage_status.
- Never count 0-row writes as successful data updates.
- Never describe PARTIAL_COVERAGE, NO_DATA, or PIT_NOT_VALIDATED as complete.
- Prompt rules explain tool semantics; facts come only from tool results and the
  evidence ledger conflict report.
""".strip()


# ── Orchestrator ──


class AgentOrchestrator:
    """Execute LLM tool loops with real-time event streaming."""

    def __init__(
        self,
        settings: Settings | None = None,
        data_lake: DataLake | None = None,
        runtime: AgentRuntime | None = None,
    ) -> None:
        resolved = settings or get_settings()
        self.runtime = runtime or build_default_runtime(resolved)
        self.settings = self.runtime.settings
        self._lake = data_lake or self.runtime.lake
        self._reports_dir = resolved.project_root / "reports" / "backtests"
        self._research_dir = resolved.project_root / "reports" / "research"
        self._approvals_dir = resolved.project_root / "approvals"

    @property
    def lake(self) -> DataLake:
        return self._lake

    def _build_registry(self) -> AgentToolRegistry:
        """Build the full AgentTool registry used by chat orchestration."""
        return self.runtime.agent_registry()

    async def execute_stream(
        self,
        message: str,
        *,
        run_id: str | None = None,
        session_id: str | None = None,
        history: list[dict[str, Any]] | None = None,
        max_rounds: int = 100,
        cancel_event: asyncio.Event | None = None,
    ) -> AsyncGenerator[OrchestratorEvent, None]:
        """Execute the LLM tool loop and yield events suitable for SSE streaming.

        If cancel_event is set, the generator checks it during the polling loop
        and yields a 'cancelled' event instead of completing normally.
        """
        rid = run_id or new_id("run")
        sid = session_id or rid
        experiment_id = new_id("exp")
        paths = PersistencePaths.from_settings(self.settings)
        ExperimentStore(paths.experiments_root, locks_root=paths.locks_root,
            quarantine_root=paths.quarantine_root / "experiments").create_experiment(
            "chat_research",
            experiment_id=experiment_id,
            hypothesis={"message": message, "session_id": sid},
            tags=["chat_research", f"session:{sid}"],
        )

        # ── Emit start ──
        yield OrchestratorEvent(
            type="run_started",
            run_id=rid,
            data={"experiment_id": experiment_id},
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
        if _requires_fresh_evidence(message):
            routing_hint += (
                "\nFresh evidence is likely needed for this request. Prefer existing "
                "conversation/tool evidence only when it directly answers the current "
                "question and is still temporally current; otherwise call relevant tools "
                "and ground the answer in the new tool results."
            )
        if _is_trade_decision_request(message):
            routing_hint += (
                "\nThis is a trade/risk decision request. Use current market data plus "
                "factor, backtest, or experiment evidence. If the context already has "
                "that evidence and it is still current, cite it explicitly; otherwise "
                "call tools such as query_bars, list_saved_factors, search_experiments, "
                "run_backtest, or evaluate_factor_candidate. If you give a buy, sell, "
                "hold, reduce, or add-position view, state clearly that it is a "
                "research-only judgement and not a live trading instruction."
            )
        if _is_data_acquisition_request(message):
            routing_hint += (
                "\nFor data acquisition or coverage-check requests, missing local bars "
                "are not by themselves a blocker. First call list_tushare_capabilities, "
                "then plan_tushare_fetch with exact ts_code values and exact field names, "
                "then call run_tushare_fetch with execute_plan=true for bounded plans "
                "that fit the tool limits. For a combination or basket, pass the full "
                "requested symbols list into the structured fetch item instead of treating "
                "another symbol's coverage as sufficient. Do not stop after a plan or ask "
                "whether to fetch when the user has asked you to get, check, retry, or "
                "verify the data. If a requested history is too long for one call, report "
                "the structured BLOCKED result or narrow to allowed windows."
            )
        if _is_large_batch_data_request(message):
            routing_hint += (
                "\nFor large-basket or bulk data pulls, use plan_tushare_fetch so the "
                "planner chooses fanout_by_symbol_range, marketwide_by_trade_date, or "
                "BLOCKED rather than hand-rolling dozens of concurrent per-symbol calls. "
                "Do not expand the user-requested date window to fetch extra history "
                "unless the user explicitly asks for that wider range. Pass the full "
                "requested symbols=[...] list to plan_tushare_fetch/run_tushare_fetch. "
                "After the batch update returns, verify the full requested symbols list "
                "with query_bars using symbols=[...] and the same requested date window. "
                "Do not validate only a sample symbol."
            )

        system_msg = "\n\n".join(
            [
                SYSTEM_PROMPT_GLOBAL,
                SYSTEM_PROMPT_DATA_RULES,
                SYSTEM_PROMPT_REPAIR_RULES,
                SYSTEM_PROMPT_VERIFICATION_RULES,
                SYSTEM_PROMPT_FINAL_ANSWER_RULES,
                (
            "You are the QMT research agent. Use tools for local facts and for "
            "multi-step research loops. You may read data, write generated research "
            "artifacts, generate candidate code in the sandbox, and run simulated "
            "backtests. You must not submit live orders, modify live config, or bypass "
            "approvals. External MCP tools may appear with the configured prefix and "
            "follow the same permission, audit, and evidence rules as native tools. "
            "For local quant research, prefer native data, factor, backtest, and report "
            "tools; do not call external MCP/web tools unless the user explicitly asks "
            "for external news/web context or native tools cannot answer the request. "
            "Tools that require human approval are not available to you as function "
            "calls. Use list_tools and describe_tool when you need to discover "
            "the available surface. Keep calling distinct useful tools until you have "
            "enough evidence to answer; avoid repeating the same tool with the same "
            "arguments. If a tool returns NOT_AVAILABLE or an error, either choose a "
            "different relevant tool or report the blocker clearly. Always respond in "
            "Chinese. Typical factor loop: list_saved_factors, list_data_catalog, "
            "create_universe_spec, validate_universe_spec, build_universe, query_universe, "
            "query_bars, create_factor_spec, generate_factor_code, run_factor_static_checks, "
            "save_factor, evaluate_factor_candidate, generate_research_report. Typical strategy "
            "loop: search_experiments, list_strategy_candidates, create_strategy_spec, "
            "generate_strategy_code, run_strategy_static_checks, save_strategy_candidate or "
            "save_strategy_spec_draft, run_backtest, generate_research_report. "
            "generate_factor_code accepts python_function as the primary path for unrestricted "
            "agent-authored research factors; if formula fallback returns NEEDS_PYTHON_FUNCTION, "
            "retry with python_function instead of narrowing the research question. Typical "
            "self-bootstrap loop: "
            "search_experiments, detect_tool_gap, create_tool_spec, generate_tool_code, "
            "generate_tool_tests, run_tool_sandbox_tests, score_tool_candidate. "
            "For multi-step tasks, create or refresh a session todo-list first "
            "with todo_set_list. Before starting a step, mark it IN_PROGRESS "
            "with todo_update_item. Mark completed steps COMPLETED, and mark "
            "externally blocked steps BLOCKED with a short note. Use todo_get_status "
            "when resuming a session or when the current todo state is unclear. "
            "When the user asks what happened in the previous/current run, what "
            "difficulties were encountered, or what tool-chain boundaries were exposed, "
            "use todo_get_status plus get_experiment_tool_calls or search_experiments "
            "when available. Prefer the current session's tool audit; do not broaden to "
            "older experiments unless the user explicitly asks for historical comparison. "
            "Mention only tools and failures observed in tool evidence; "
            "do not claim run_tushare_fetch, data fetches, or code changes happened "
            "unless the tool audit or current tool result shows them. "
            "When a tool returns actual_data_end or data_freshness, distinguish the "
            "local latest data date from the requested end date and from the latest "
            "market trading day; do not call data complete through today when "
            "data_freshness is stale_vs_requested_end. A remote data dry_run with "
            "status=planned is only a plan and does not mean data was fetched or gaps "
            "were proven harmless. For current position, buy/sell, next-trading-day, "
            "or risk-decision questions, make fresh tool calls for latest bars and "
            "relevant factor/backtest evidence before answering, even if prior "
            "conversation history contains recent-looking data unless that history already "
            "contains the exact current evidence you need. When the user asks to retry, "
            "retest, rerun, verify again, or check whether a previous tool problem was fixed, "
            "prefer fresh tool calls unless prior tool results already prove the answer. "
            "For buy, sell, hold, position, and risk-decision answers, do not infer current "
            "factor signals from memory alone; use or cite fresh market data plus factor, "
            "backtest, or experiment evidence, and label the conclusion as research-only "
            "rather than a live order. Do not claim that code or tools were fixed unless "
            "you actually generated or changed code, or a tool result explicitly reports "
            "an update. If you only changed tool arguments or reran checks, describe it "
            "as rerun/verified. For remote data tools, start_date/end_date are request "
            "bounds; actual_data_end or coverage_end_date is the local data coverage "
            "date. Do not say local data covers a non-trading requested_end_date unless "
            "actual_data_end also equals that date. If a remote data plan "
            "returns requires_trade_calendar_validation=true or "
            "missing_ranges_are_calendar_days=true, do not describe those ranges as "
            "weekends or holidays unless another tool result proves that. Treat "
            "CALENDAR_VALIDATION_REQUIRED as an explicit blocker for any claim that "
            "data gaps are harmless. If query_bars or run_tushare_fetch returns "
            "status=PARTIAL_COVERAGE, missing_symbols, or stale_symbols, do not claim "
            "the basket data is complete. After filling data gaps, verify coverage with "
            "query_bars using the same symbols=[...] list and the same date window; "
            "only treat the basket as complete when status=OK and missing_symbols and "
            "stale_symbols are empty. For any factor or strategy research, first define "
            "or resolve an explicit universe. Use create_universe_spec, "
            "validate_universe_spec, build_universe, save_universe_spec, list_universes, "
            "inspect_universe, and query_universe. Do not hand-write symbols unless the "
            "user explicitly supplied them or a universe tool returned validated symbols. "
            "Generated universes are research-only and not live-trading-approved. For "
            "rolling universes, disclose per-date symbol resolution and empty-date "
            "diagnostics. If "
            "query_fundamentals_pit returns NO_DATA/PARTIAL_COVERAGE or INVALID_REQUEST "
            "because the date window is too large, call list_tushare_capabilities, then "
            "plan_tushare_fetch for the needed fundamental endpoint, then run_tushare_fetch "
            "with execute_plan=true when live update is allowed and verify with "
            "query_fundamentals_pit. If query_macro_series_pit returns INVALID_REQUEST, "
            "fix the request arguments first, usually by using list_tushare_capabilities; "
            "do not call run_tushare_fetch for an unknown dataset. If "
            "query_macro_series_pit returns NO_DATA for a known dataset, use its "
            "repair_action or list_tushare_capabilities and plan_tushare_fetch for "
            "the exact macro endpoint; when live update is allowed execute with "
            "run_tushare_fetch execute_plan=true and verify with "
            "query_macro_series_pit. Do not infer macro timing signals from missing or "
            "unknown macro datasets. If "
            "generate_factor_code returns STATIC_CHECK_FAILED, or "
            "run_factor_static_checks returns semantic_status=FAILED, do not save, "
            "evaluate, rank, or recommend that generated factor. "
            "If a tool returns BLOCKED, INVALID_REQUEST, NO_DATA, PARTIAL_COVERAGE, "
            "SAMPLE_TEST_FAILED, STATIC_CHECK_FAILED, or semantic_status=FAILED and "
            "the result includes next_repair_tool or suggested_repair, treat that as "
            "the next concrete repair step unless the user goal explicitly does not "
            "need that path. If you cannot execute the repair, use todo_update_item to "
            "mark the step BLOCKED and include the tool name, reason, missing fields, "
            "and next_repair_tool in the final answer. MISSING_FACTOR_INPUTS means "
            "the factor inputs are absent from local data; do not call the factor "
            "invalid until the missing inputs are fetched or the exact blocker is "
            "reported. For fundamental factors such as dividend_yield, pb_rank, or "
            "roe_rank, missing dv_ttm, pb, or roe should route through "
            "list_tushare_capabilities, plan_tushare_fetch, then a scoped "
            "run_tushare_fetch when allowed, followed by query_fundamentals_pit "
            "verification. Before building strategy_spec.factors or selected_factors, "
            "call list_saved_factors or describe_factor to confirm the exact factor_id. "
            "Every strategy factor leg must use factor_id; never use factor_name as a "
            "strategy factor leg field. Keep the final answer "
            "at the same abstraction level as the user's question: if the user asks for "
            "strategies, factor tests are intermediate evidence and must not be presented "
            "as strategy results. Do not claim that code was generated, static checks "
            "passed, candidates were saved, data was refreshed, or a specific universe "
            "was used unless a tool result or tool audit explicitly shows it. Treat "
            "adapter_limitations, warnings, diagnostics FAIL/BLOCKED/NOT_COMPUTED, "
            "missing data, and REVIEW_REQUIRED as material limitations. The system "
            "preserves your raw final answer and separately reports evidence-output "
            "conflicts; do not hide or smooth over tool statuses to make the answer "
            "look cleaner. Do not claim a token or external API is unavailable "
            "unless a tool returned NOT_CONFIGURED, ENV_BLOCKED, or an explicit "
            "upstream/API error; if a tool returns BLOCKED for missing ts_code or "
            "unsupported basket live fill, call it a tool scope or adapter limitation. "
            "Do not blame "
            "replay, validation, or test protocols for data or tool limitations; "
            "attribute blockers to the observed tool status, missing inputs, adapter "
            "capability, external API state, or data coverage."
            " For data acquisition or coverage-check requests, do not stop after a "
            "dry_run plan or ask whether to fetch when local data is stale or missing; "
            "perform the tool-supported scoped remote update yourself with "
            "run_tushare_fetch execute_plan=true; if the tool returns "
            "BLOCKED/INVALID_REQUEST/NO_DATA, "
            "carry that structured blocker into the final answer instead of inventing "
            "a protocol limit."
            " For large-basket or bulk data pulls, pass the full symbols=[...] list "
            "for coverage checks and obey the update tool's live-execution scope. "
            "If a live basket fill is not supported, report that adapter/tool "
            "limitation explicitly. Do not "
            "expand the user-requested date window. Do not validate only a sample symbol."
                ),
                routing_hint.strip(),
            ]
        )

        tools = self.runtime.llm_tools(
            run_id=rid,
            session_id=sid,
            experiment_id=experiment_id,
        )

        messages: list[dict[str, Any]] = [{"role": "system", "content": system_msg}]
        messages.extend(_conversation_history(history or [], current_message=message))
        messages.append({"role": "user", "content": message})

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
                fallback_tool_results = [
                    evt
                    for evt in stream_buffer
                    if evt.__class__.__name__ == "ToolResult"
                ]
                if fallback_tool_results:
                    fallback_message = _stream_error_fallback_message(
                        stream_error_msg,
                        fallback_tool_results,
                    )
                    yield OrchestratorEvent(
                        type="error",
                        run_id=rid,
                        message=(
                            f"LLM streaming error: {stream_error_msg}; "
                            "returning fallback summary from completed tool evidence."
                        ),
                        data={
                            "error": stream_error_msg,
                            "fallback": True,
                            "experiment_id": experiment_id,
                        },
                    )
                    yield OrchestratorEvent(
                        type="final_message",
                        run_id=rid,
                        message=fallback_message,
                        data={
                            "content": fallback_message,
                            "phase": "final",
                            "fallback": True,
                            "stream_error": stream_error_msg,
                            "experiment_id": experiment_id,
                        },
                    )
                    yield OrchestratorEvent(
                        type="done",
                        run_id=rid,
                        message="Completed with stream error fallback.",
                        data={
                            "experiment_id": experiment_id,
                            "completed_with_stream_error": True,
                            "fallback": True,
                            "tool_calls_count": len(fallback_tool_results),
                        },
                    )
                    return
                yield OrchestratorEvent(
                    type="error",
                    run_id=rid,
                    message=f"LLM streaming error: {stream_error_msg}",
                    data={"error": stream_error_msg},
                )
                return

            terminal_errors = [
                evt
                for evt in stream_buffer
                if evt.__class__.__name__ in {"SafetyCapHit", "LoopError"}
            ]
            if terminal_errors:
                return

        # ── Count tool calls ──
        tcount = sum(
            1 for e in stream_buffer if e.__class__.__name__ == "ToolResult"
        )
        ledger = _ledger_from_stream(rid, stream_buffer)
        final_answer_raw = _final_answer_from_stream(stream_buffer)
        evidence_report = ledger.report()
        if final_answer_raw is not None:
            conflict_report = ledger.final_answer_conflict_report(final_answer_raw)
            if conflict_report["has_conflict"]:
                yield OrchestratorEvent(
                    type="evidence_conflict_report",
                    run_id=rid,
                    message="Evidence conflicts detected after raw final answer.",
                    data={
                        "experiment_id": experiment_id,
                        "evidence_conflict_report": conflict_report,
                    },
                )
        yield OrchestratorEvent(
            type="done",
            run_id=rid,
            message=_done_message(evidence_report),
            data={
                "experiment_id": experiment_id,
                "tool_calls_count": tcount,
                "evidence_report": evidence_report,
            },
        )


# ── Stream event converter ──


def _stream_error_fallback_message(error: str, tool_results: list[Any]) -> str:
    lines = [
        f"LLM stream interrupted before final answer: {error}",
        "Fallback summary from completed tool evidence only:",
    ]
    for index, evt in enumerate(tool_results, start=1):
        tool_name = str(getattr(evt, "tool_name", "unknown_tool"))
        result = getattr(evt, "result", None)
        lines.append(f"- {index}. {tool_name}: {_fallback_result_summary(result)}")
    lines.append(
        "Unfinished: the model final response was not received; "
        "this fallback does not add conclusions beyond completed tool evidence."
    )
    return "\n".join(lines)


def _fallback_result_summary(result: Any) -> str:
    if not isinstance(result, dict):
        return _preview(result, max_len=300)

    fields: list[str] = []
    for key in (
        "status",
        "reason",
        "run_id",
        "strategy_id",
        "factor_id",
        "report_path",
        "next_repair_tool",
    ):
        value = result.get(key)
        if value not in (None, "", [], {}):
            fields.append(f"{key}={value}")

    metrics = result.get("metrics")
    if isinstance(metrics, dict):
        metric_parts = [
            f"{key}={value}"
            for key, value in metrics.items()
            if value not in (None, "", [], {})
        ]
        if metric_parts:
            fields.append("metrics={" + ", ".join(metric_parts[:6]) + "}")

    issues = result.get("issues")
    if isinstance(issues, list) and issues:
        fields.append("issues=" + "; ".join(str(item) for item in issues[:3]))

    if not fields:
        fields.append(_preview(result, max_len=300))
    return ", ".join(fields)


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
                data={
                    "token": evt.content,
                    "phase": getattr(evt, "phase", "draft"),
                    "experiment_id": experiment_id,
                },
            )
        ]

    if cls_name == "FinalMessage":
        return [
            OrchestratorEvent(
                type="final_message",
                run_id=run_id,
                message=evt.content,
                data={
                    "content": evt.content,
                    "phase": "final",
                    "experiment_id": experiment_id,
                },
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
        result_id = _result_id(evt.result)
        status = _tool_status_display(evt.result)
        events = [
            OrchestratorEvent(
                type="tool_done",
                run_id=run_id,
                message=f"Tool: {evt.tool_name} {status['symbol']}",
                data={
                    "tool_name": evt.tool_name,
                    "result_preview": preview,
                    "result_id": result_id,
                    "execution_status": status["execution_status"],
                    "domain_status": status["domain_status"],
                    "evidence_status": status["evidence_status"],
                    "recommendation_status": status["recommendation_status"],
                    "raw_status": status["raw_status"],
                    "diagnostic_status": status["diagnostic_status"],
                    "experiment_id": experiment_id,
                },
            )
        ]
        todo_event = _todo_status_event(evt.result, run_id, experiment_id)
        if todo_event is not None:
            events.append(todo_event)
        return events

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
                type="error",
                run_id=run_id,
                message=evt.message,
                data={"error": evt.message},
            )
        ]

    return []


def _ledger_from_stream(run_id: str, stream_buffer: list[Any]) -> EvidenceLedger:
    ledger = EvidenceLedger(run_id=run_id)
    for evt in stream_buffer:
        if evt.__class__.__name__ == "ToolResult":
            ledger.record_tool_result(
                str(getattr(evt, "tool_name", "unknown_tool")),
                getattr(evt, "result", None),
            )
    return ledger


def _final_answer_from_stream(stream_buffer: list[Any]) -> str | None:
    for evt in reversed(stream_buffer):
        if evt.__class__.__name__ == "FinalMessage":
            return str(getattr(evt, "content", ""))
    return None


def _done_message(evidence_report: dict[str, Any]) -> str:
    summary = evidence_report.get("summary", {})
    if not isinstance(summary, dict):
        return "Run finished: all tool calls returned, evidence state unavailable."
    if summary.get("invalid_count", 0):
        return "Run finished: all tool calls returned, evidence invalid."
    if summary.get("blocked_count", 0) or evidence_report.get("blockers"):
        return "Run finished: all tool calls returned, evidence has blockers."
    if summary.get("unknown_count", 0):
        return "Run finished: all tool calls returned, evidence unknown."
    if summary.get("weak_count", 0) or summary.get("incomplete_count", 0):
        return "Run finished: all tool calls returned, evidence mixed."
    return "Run finished: all tool calls returned."


def _tool_status_display(result: Any) -> dict[str, Any]:
    if not isinstance(result, dict):
        return {
            "symbol": "?",
            "execution_status": "UNKNOWN",
            "domain_status": "UNKNOWN",
            "evidence_status": "UNKNOWN",
            "recommendation_status": "UNKNOWN",
            "raw_status": None,
            "diagnostic_status": None,
        }
    icon = status_icon(result)
    symbol = {
        "ok": "✓",
        "warning": "⚠",
        "failed": "✗",
        "blocked": "⛔",
        "unknown": "?",
        "x": "✗",
    }.get(icon, "?")
    return {
        "symbol": symbol,
        "execution_status": result.get("execution_status", "UNKNOWN"),
        "domain_status": result.get("domain_status", "UNKNOWN"),
        "evidence_status": result.get("evidence_status", "UNKNOWN"),
        "recommendation_status": result.get("recommendation_status", "UNKNOWN"),
        "raw_status": result.get("raw_status"),
        "diagnostic_status": result.get("diagnostic_status"),
    }


def _conversation_history(
    history: list[dict[str, Any]],
    *,
    current_message: str,
    max_turns: int = 12,
) -> list[dict[str, str]]:
    natural_messages: list[dict[str, str]] = []
    for raw in history:
        role = str(raw.get("role", ""))
        if role not in {"user", "assistant"}:
            continue
        content = str(raw.get("content", "")).strip()
        if not content:
            continue
        natural_messages.append({"role": role, "content": content})

    if natural_messages and natural_messages[-1] == {
        "role": "user",
        "content": current_message.strip(),
    }:
        natural_messages = natural_messages[:-1]

    return natural_messages[-max_turns * 2 :]


def _requires_fresh_evidence(message: str) -> bool:
    normalized = message.strip().lower()
    if not normalized:
        return False
    patterns = [
        r"再试试",
        r"重新(验证|测试|检查|跑|运行|评估)",
        r"(修复|改完|修好|好了).{0,12}(再|重新|试|验证|测试|检查)",
        r"(tool|工具).{0,12}(出错|错误|修复|问题)",
        r"(retry|retest|rerun|run again|try again|verify again|check again)",
        r"(buy|sell|position|risk|next trading day|today|tomorrow)",
        r"(买|卖|仓位|持仓|风险|下个交易日|今天|明天)",
    ]
    return any(re.search(pattern, normalized, flags=re.IGNORECASE) for pattern in patterns)


def _is_trade_decision_request(message: str) -> bool:
    normalized = message.strip().lower()
    if not normalized:
        return False
    return any(
        re.search(pattern, normalized, flags=re.IGNORECASE)
        for pattern in [
            r"(买|卖|减仓|加仓|清仓|持有|持仓|仓位|下个交易日|交易决策)",
            r"(buy|sell|hold|reduce|add position|position|trade decision)",
        ]
    )


def _is_data_acquisition_request(message: str) -> bool:
    normalized = message.strip().lower()
    if not normalized:
        return False
    return any(
        re.search(pattern, normalized, flags=re.IGNORECASE)
        for pattern in [
            r"(获取|拉取|同步|补齐|更新|下载).{0,40}(数据|行情|日线)",
            r"(检查|确认|验证|看看).{0,40}(获取|数据|行情|覆盖|缺口|新鲜)",
            r"(自上市以来|上市以来|发行以来)",
            r"(fetch|sync|update|download|get).{0,40}(data|bars|quotes)",
            r"(coverage|freshness|missing|gap).{0,40}(data|bars|quotes)",
        ]
    )


def _is_large_batch_data_request(message: str) -> bool:
    normalized = message.strip().lower()
    if not normalized:
        return False
    return any(
        re.search(pattern, normalized, flags=re.IGNORECASE)
        for pattern in [
            r"(大批量|批量|几十个|数十个).{0,40}(标的|股票|证券|数据|行情|日线|拉取|同步)",
            r"(标的|股票|证券).{0,20}(大批量|批量|几十个|数十个).{0,40}(数据|行情|日线|拉取|同步)",
            r"(large|big|bulk).{0,20}(basket|batch|symbol|symbols|ticker|tickers).{0,40}(data|bars|pull|fetch|sync)",
            r"(basket|batch).{0,20}(50|[2-9][0-9]).{0,40}(symbol|symbols|ticker|tickers)",
        ]
    )


def _result_id(result: Any) -> str:
    payload = json.dumps(result, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _todo_status_event(
    result: Any,
    run_id: str,
    experiment_id: str,
) -> OrchestratorEvent | None:
    if not isinstance(result, dict):
        return None
    state = result.get("todo_state")
    if not isinstance(state, dict):
        return None
    data = {
        "session_id": state.get("session_id"),
        "items": state.get("items", []),
        "summary": state.get("summary", {}),
        "active_item": state.get("active_item"),
        "goal": state.get("goal"),
        "updated_at": state.get("updated_at"),
        "experiment_id": experiment_id,
    }
    summary = data["summary"] if isinstance(data["summary"], dict) else {}
    return OrchestratorEvent(
        type="todo_status",
        run_id=run_id,
        message=(
            f"Todo status: {summary.get('completed', 0)}/"
            f"{summary.get('total', 0)} completed"
        ),
        data=data,
    )


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
