"""Chat page — natural conversation with real LLM orchestration via SSE.

Each browser tab/session gets its own AgentOrchestrator instance.
Cards accumulate across turns so history is preserved.
"""

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


def _fill_prompt(message_input: ui.textarea, text: str) -> None:
    message_input.value = text


async def _send(
    orchestrator: AgentOrchestrator,
    transcript_col: ui.column,
    message_input: ui.textarea,
    plan_card: ui.card,
    progress_card: ui.card,
) -> None:
    content = (message_input.value or "").strip()
    if not content:
        ui.notify("Enter a message first.", type="warning")
        return

    message_input.value = ""

    # ── User message card ──
    user_card = ui.card().classes("w-full bg-white border p-3")
    with user_card:
        ui.markdown(f"**🧑 You**  \n{content}")
    user_card.move(transcript_col)

    # ── Step 1: Route intent ──
    decision = agent_router.route(content)
    intent = decision.intent.value
    confidence = decision.confidence

    plan_html = (
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
        ui.markdown(f"### 🤖 Agent Plan\n\n{plan_html}")
    plan_card.visible = True

    # ── Step 2: Run orchestration ──
    run_id = new_id("run")

    assistant_card: ui.card | None = None
    assistant_md: ui.markdown | None = None
    token_buf: list[str] = []
    progress_label_ref: list[ui.label | None] = [None]
    need_new_card: bool = False

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
                info_card = ui.card().classes(
                    "w-full bg-gray-50 p-2 text-xs text-gray-500"
                )
                with info_card:
                    ui.label(
                        f"**Run** `{run_id[:8]}` | "
                        f"**Exp** `{exp_id}` | **Intent** `{intent}`"
                    )
                info_card.move(transcript_col)

            elif etype == "progress":
                if emsg and progress_label_ref[0] is not None:
                    progress_label_ref[0].set_text(emsg)

            elif etype == "token":
                if assistant_card is None or need_new_card:
                    assistant_card = ui.card().classes(
                        "w-full bg-blue-50 border p-3"
                    )
                    with assistant_card:
                        ui.markdown("**🤖 Assistant**").classes(
                            "text-sm font-semibold text-blue-800"
                        )
                        assistant_md = ui.markdown("").classes(
                            "text-sm text-blue-900"
                        )
                    assistant_card.move(transcript_col)
                    token_buf = []
                    need_new_card = False
                token_buf.append(emsg)
                if assistant_md is not None:
                    assistant_md.set_content("".join(token_buf))

            elif etype == "tool_start":
                tool_name = edata.get("tool_name", "")
                progress_card.clear()
                with progress_card:
                    with ui.row().classes("items-center gap-2"):
                        ui.spinner(size="sm")
                        lbl = ui.label(f"Executing: `{tool_name}`").classes("text-sm")
                        progress_label_ref[0] = lbl
                progress_card.visible = True

                tool_card = ui.card().classes("w-full bg-gray-50 border p-2 text-xs")
                with tool_card:
                    ui.markdown(f"🔧 **Calling:** `{tool_name}`")
                tool_card.move(transcript_col)
                need_new_card = True

            elif etype == "tool_args":
                args = edata.get("arguments", {})
                import json as _json
                args_str = _json.dumps(args, ensure_ascii=False, default=str)
                args_card = ui.card().classes("w-full bg-gray-50 border p-2 text-xs")
                with args_card:
                    ui.markdown(f"```json\n{args_str[:500]}\n```")
                args_card.move(transcript_col)

            elif etype == "tool_done":
                preview = edata.get("result_preview", "")
                progress_card.visible = False
                result_card = ui.card().classes(
                    "w-full bg-gray-50 border p-2 text-xs"
                )
                with result_card:
                    ui.markdown(f"✅ **Result:** `{preview}`")
                result_card.move(transcript_col)

            elif etype == "done":
                progress_card.visible = False
                plan_card.visible = False
                tool_count = edata.get("tool_calls_count", 0)
                done_card = ui.card().classes("w-full bg-green-50 border p-2")
                with done_card:
                    ui.markdown(
                        f"**✅ Done** — {tool_count} tool call(s) completed."
                    )
                done_card.move(transcript_col)

            elif etype == "error":
                progress_card.visible = False
                plan_card.visible = False
                err_card = ui.card().classes(
                    "w-full bg-red-50 border border-red-300 p-3"
                )
                with err_card:
                    ui.icon("error", color="red").classes("inline")
                    ui.markdown(f"**❌ Error**  \n{emsg}")
                err_card.move(transcript_col)

    except Exception as exc:
        progress_card.visible = False
        plan_card.visible = False
        err_card = ui.card().classes(
            "w-full bg-red-50 border border-red-300 p-3"
        )
        with err_card:
            ui.markdown(f"**❌ Orchestration failed:** {exc}")
        err_card.move(transcript_col)


def register() -> None:
    @ui.page("/")
    def chat_page() -> None:
        shell("Chat")

        # ── Per-session orchestrator (isolated from other tabs/users) ──
        orchestrator = AgentOrchestrator(settings=get_settings())

        with ui.column().classes("w-full gap-2"):
            ui.label("QMT Agent Studio").classes("text-2xl font-semibold")
            ui.label(
                "Ask anything about quantitative research — "
                "Agent routes and executes automatically."
            ).classes("text-sm text-gray-500 mb-2")

        # ── Session transcript ──
        transcript_col = ui.column().classes("w-full gap-2")

        # ── Plan card ──
        plan_card = ui.card().classes("w-full bg-blue-50 p-3")
        plan_card.visible = False

        # ── Progress card ──
        progress_card = ui.card().classes("w-full bg-green-50 p-3")
        with progress_card:
            with ui.row().classes("items-center gap-2"):
                ui.spinner(size="sm")
                ui.label("").classes("text-sm")
        progress_card.visible = False

        # ── Input row ──
        with ui.row().classes("w-full items-end gap-2"):
            message = (
                ui.textarea("Type your research question...")
                .classes("grow")
                .props("autogrow rows=2 outlined")
            )
            ui.button(
                "Send",
                on_click=lambda: _send(
                    orchestrator,
                    transcript_col,
                    message,
                    plan_card,
                    progress_card,
                ),
            ).props("color=primary")

        # ── Advanced panel ──
        with ui.expansion("Advanced", icon="tune").classes("w-full"):
            with ui.row().classes("gap-4"):
                ui.select(
                    ["auto", "stock", "etf", "stock_etf"],
                    value="auto",
                    label="Universe",
                ).classes("w-40")
                with ui.row().classes("gap-2"):
                    ui.input("Start Date", value="").classes("w-36").props(
                        "placeholder=auto"
                    )
                    ui.input("End Date", value="").classes("w-36").props(
                        "placeholder=auto"
                    )
                ui.select(
                    ["balanced", "fast", "thorough"],
                    value="balanced",
                    label="Budget",
                ).classes("w-36")

        # ── Suggested prompts ──
        with ui.expansion("Suggested prompts", icon="lightbulb").classes("w-full"):
            with ui.row().classes("flex-wrap gap-2"):
                for p in SUGGESTED_PROMPTS:
                    ui.chip(
                        p, on_click=lambda _, text=p: _fill_prompt(message, text)
                    )

        # ── LLM status bar ──
        with ui.row().classes("items-center gap-2 mt-2 text-xs text-gray-400"):
            if orchestrator.settings.deepseek_api_key:
                ui.icon("check_circle", size="xs", color="green")
                ui.label(
                    f"DeepSeek connected ({orchestrator.settings.deepseek_model})"
                )
            else:
                ui.icon("warning", size="xs", color="orange")
                ui.label("DeepSeek not configured — stub mode only")
