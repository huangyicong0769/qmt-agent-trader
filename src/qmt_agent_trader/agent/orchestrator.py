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

from qmt_agent_trader.agent.experiment_store import ExperimentStore
from qmt_agent_trader.agent.llm_client import (
    DeepSeekClient,
)
from qmt_agent_trader.agent.runtime import AgentRuntime, build_default_runtime
from qmt_agent_trader.agent.tool_registry import AgentToolRegistry
from qmt_agent_trader.core.config import Settings, get_settings
from qmt_agent_trader.core.ids import new_id
from qmt_agent_trader.data.storage import DataLake

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
        ExperimentStore(self.settings.resolved_data_dir / "experiments").create_experiment(
            "chat_research",
            experiment_id=experiment_id,
            hypothesis={"message": message},
            tags=["chat_research"],
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
                "are not by themselves a blocker. Use run_remote_data_update to fetch "
                "remote data when the local lake is stale or incomplete. Start with "
                "dry_run=true to identify gaps, then call dry_run=false for the bounded "
                "missing ranges that fit the tool limits. For a combination or basket, "
                "loop over each requested symbol instead of treating another symbol's "
                "coverage as sufficient. Do not stop after a dry_run plan or ask "
                "whether to fetch when the user has asked you to get, check, retry, "
                "or verify the data. If a requested history is too long for one call, "
                "split it into allowed windows or clearly state the remaining bounded "
                "range that was not fetched."
            )
        if _is_large_batch_data_request(message):
            routing_hint += (
                "\nFor large-basket or bulk data pulls, prefer one batch or market-wide "
                "remote update without ts_code when the requested date range is bounded "
                "and all requested securities share the same asset type. Do not fan out "
                "dozens of concurrent per-symbol run_remote_data_update calls after a "
                "batch update. Do not expand the user-requested date window to fetch "
                "extra history unless the user explicitly asks for that wider range. "
                "Pass the full requested symbols=[...] list to run_remote_data_update "
                "so the dry-run and live update can detect basket member gaps. "
                "After the batch update returns, verify the full requested symbols list "
                "with query_bars using symbols=[...] and the same requested date window. "
                "Do not validate only a sample symbol. Inspect which symbols, if any, "
                "are still missing. Only then run targeted per-symbol updates for the "
                "explicit missing subset."
            )

        system_msg = (
            "You are the QMT research agent. Use tools for local facts and for "
            "multi-step research loops. You may read data, write generated research "
            "artifacts, generate candidate code in the sandbox, and run simulated "
            "backtests. You must not submit live orders, modify live config, or bypass "
            "approvals. External MCP tools may appear with the configured prefix and "
            "follow the same permission, audit, and evidence rules as native tools. "
            "Tools that require human approval are not available to you as function "
            "calls. Use list_tools and describe_tool when you need to discover "
            "the available surface. Keep calling distinct useful tools until you have "
            "enough evidence to answer; avoid repeating the same tool with the same "
            "arguments. If a tool returns NOT_AVAILABLE or an error, either choose a "
            "different relevant tool or report the blocker clearly. Always respond in "
            "Chinese. Typical factor loop: list_saved_factors, list_data_catalog, "
            "query_universe/query_bars, create_factor_spec, generate_factor_code, "
            "run_factor_static_checks, "
            "save_factor, evaluate_factor_candidate, generate_research_report. Typical strategy "
            "loop: search_experiments, list_strategy_candidates, create_strategy_spec, "
            "generate_strategy_code, "
            "run_backtest, generate_research_report. Typical self-bootstrap loop: "
            "search_experiments, detect_tool_gap, create_tool_spec, generate_tool_code, "
            "generate_tool_tests, run_tool_sandbox_tests, score_tool_candidate. "
            "For multi-step tasks, create or refresh a session todo-list first "
            "with todo_set_list. Before starting a step, mark it IN_PROGRESS "
            "with todo_update_item. Mark completed steps COMPLETED, and mark "
            "externally blocked steps BLOCKED with a short note. Use todo_get_status "
            "when resuming a session or when the current todo state is unclear. "
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
            "data gaps are harmless. If query_bars or run_remote_data_update returns "
            "status=PARTIAL_COVERAGE, missing_symbols, or stale_symbols, do not claim "
            "the basket data is complete. After filling data gaps, verify coverage with "
            "query_bars using the same symbols=[...] list and the same date window; "
            "only treat the basket as complete when status=OK and missing_symbols and "
            "stale_symbols are empty."
            " For data acquisition or coverage-check requests, do not stop after a "
            "dry_run plan or ask whether to fetch when local data is stale or missing; "
            "perform the bounded remote update yourself with dry_run=false, and for "
            "baskets loop over each requested symbol."
            " For large-basket or bulk data pulls, prefer one batch or market-wide "
            "remote update without ts_code while passing the full symbols=[...] list, "
            "then verify the full requested symbols list with query_bars before "
            "considering any per-symbol retries. Do not "
            "expand the user-requested date window. Do not validate only a sample symbol."
            + routing_hint
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
        yield OrchestratorEvent(
            type="done",
            run_id=rid,
            message="Run completed successfully.",
            data={
                "experiment_id": experiment_id,
                "tool_calls_count": tcount,
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
        events = [
            OrchestratorEvent(
                type="tool_done",
                run_id=run_id,
                message=f"Tool: {evt.tool_name} ✓",
                data={
                    "tool_name": evt.tool_name,
                    "result_preview": preview,
                    "result_id": result_id,
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
