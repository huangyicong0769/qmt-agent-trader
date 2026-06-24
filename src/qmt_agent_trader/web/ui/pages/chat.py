"""Chat page — natural conversation with real LLM orchestration via SSE."""

from __future__ import annotations

from nicegui import ui

from qmt_agent_trader.agent.orchestrator import AgentOrchestrator
from qmt_agent_trader.agent.router import agent_router
from qmt_agent_trader.core.config import get_settings
from qmt_agent_trader.core.ids import new_id
from qmt_agent_trader.web.ui.layout import shell

SUGGESTED_PROMPTS = [
    "帮我发现几个适合A股个股和ETF的低波动高胜率因子，并自动跑初步验证。",
    "列出当前数据湖中所有可用的因子，并验证 momentum_20d 因子。",
    "基于最近有效的候选因子，写一个日频轮动策略并回测。",
    "看看最近失败的实验，判断是不是缺少某个工具。",
    "解释一下上一个回测为什么收益高但回撤也大。",
]

_orchestrator: AgentOrchestrator | None = None


def _get_orchestrator() -> AgentOrchestrator:
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = AgentOrchestrator(settings=get_settings())
    return _orchestrator


def register() -> None:
    @ui.page("/")
    def chat_page() -> None:
        shell("Chat")

        ui.label("QMT Agent Studio").classes("text-2xl font-semibold")
        ui.label(
            "Ask anything about quantitative research — Agent routes and executes automatically."
        ).classes("text-sm text-gray-500 mb-4")

        with ui.column().classes("w-full gap-4"):
            transcript = ui.markdown(
                "**🤖 Assistant:** Ready for research. Type your question and press Send."
            ).classes("min-h-[300px] p-4 border rounded-lg bg-gray-50")

            plan_card = ui.card().classes("w-full bg-blue-50 p-4")
            plan_card.visible = False

            progress_card = ui.card().classes("w-full bg-green-50 p-4")
            progress_card.visible = False

            with ui.row().classes("w-full items-end gap-2"):
                message = (
                    ui.textarea("Type your research question...")
                    .classes("grow")
                    .props("autogrow rows=2")
                )
                with ui.column().classes("gap-1"):
                    ui.button(
                        "Send",
                        on_click=lambda: _send(transcript, message, plan_card, progress_card),
                    ).props("color=primary")

            with ui.expansion("Advanced", icon="tune").classes("w-full"):
                with ui.row().classes("gap-4"):
                    ui.select(
                        ["auto", "stock", "etf", "stock_etf"],
                        value="auto",
                        label="Universe",
                    ).classes("w-40")
                    with ui.row().classes("gap-2"):
                        ui.input("Start Date", value="").classes("w-36").props("placeholder=auto")
                        ui.input("End Date", value="").classes("w-36").props("placeholder=auto")
                    ui.select(
                        ["balanced", "fast", "thorough"],
                        value="balanced",
                        label="Budget",
                    ).classes("w-36")

        with ui.expansion("Suggested prompts", icon="lightbulb").classes("w-full mt-4"):
            for p in SUGGESTED_PROMPTS:
                ui.chip(p, on_click=lambda _, text=p: _fill_prompt(message, text))

        # ── LLM status bar ──
        with ui.row().classes("items-center gap-2 mt-2 text-xs text-gray-400"):
            orch = _get_orchestrator()
            if orch.settings.deepseek_api_key:
                ui.icon("check_circle", size="xs", color="green")
                ui.label(f"DeepSeek connected ({orch.settings.deepseek_model})")
            else:
                ui.icon("warning", size="xs", color="orange")
                ui.label("DeepSeek not configured — stub mode only")


def _fill_prompt(message_input: ui.textarea, text: str) -> None:
    message_input.value = text


async def _send(
    transcript: ui.markdown,
    message_input: ui.textarea,
    plan_card: ui.card,
    progress_card: ui.card,
) -> None:
    content = (message_input.value or "").strip()
    if not content:
        ui.notify("Enter a message first.", type="warning")
        return

    # Clear state
    message_input.value = ""
    lines: list[str] = [f"**🧑 You:** {content}", ""]
    plan_card.visible = False
    progress_card.visible = False

    # ── Step 1: Route intent ──
    decision = agent_router.route(content)
    intent = decision.intent.value
    confidence = decision.confidence

    plan_html = (
        f"### 🤖 Agent Plan\n\n"
        f"| | |\n|---|---|\n"
        f"| **Intent** | `{intent}` |\n"
        f"| **Confidence** | {confidence:.0%} |\n"
    )
    if decision.proposed_workflow:
        plan_html += f"| **Workflow** | `{decision.proposed_workflow}` |\n"
    plan_html += f"\n**Rationale:** {decision.rationale}"
    if decision.required_tools:
        plan_html += (
            f"\n\n**Tools:** {', '.join(decision.required_tools[:8])}"
            f"{'…' if len(decision.required_tools) > 8 else ''}"
        )

    plan_card.clear()
    with plan_card:
        ui.markdown(plan_html)
    plan_card.visible = True

    lines.append(f"**🔍 Intent:** `{intent}` ({confidence:.0%})")
    lines.append(f"_{decision.rationale}_")
    transcript.set_content("\n".join(lines))

    # ── Step 2: Run orchestration ──
    orchestrator = _get_orchestrator()
    run_id = new_id("run")

    progress_card.clear()
    with progress_card:
        ui.spinner(size="sm")
        ui.label("Running LLM orchestration...").classes("text-sm")
    progress_card.visible = True

    try:
        async for event in orchestrator.execute_stream(
            message=content,
            routing=decision,
            run_id=run_id,
        ):
            etype = event.type
            emsg = event.message
            edata = event.data

            if etype == "run_started":
                exp_id = edata.get("experiment_id", "?")
                lines.append(f"**🚀 Started** — experiment `{exp_id}`")

            elif etype == "progress":
                if emsg:
                    lines.append(f"**🔄** {emsg}")
                    progress_card.clear()
                    with progress_card:
                        ui.spinner(size="sm")
                        ui.label(emsg).classes("text-sm")

            elif etype == "tool_done":
                tool_name = edata.get("tool_name", "")
                idx = edata.get("index", 0)
                total = edata.get("total", 0)
                lines.append(f"**🔧 [{idx}/{total}]** `{tool_name}`")

            elif etype == "llm_message":
                lines.append("")
                lines.append("**🤖 Assistant:**")
                lines.append(emsg)

            elif etype == "done":
                progress_card.visible = False
                plan_card.visible = False
                tool_count = edata.get("tool_calls_count", 0)
                lines.append("")
                lines.append(f"**✅ Done** — {tool_count} tool call(s) completed.")

            elif etype == "error":
                progress_card.visible = False
                lines.append(f"**❌ Error:** {emsg}")

            transcript.set_content("\n".join(lines))

    except Exception as exc:
        progress_card.visible = False
        lines.append(f"**❌ Orchestration failed:** {exc}")
        transcript.set_content("\n".join(lines))
