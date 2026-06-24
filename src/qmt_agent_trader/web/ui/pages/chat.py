"""Chat page — Codex-style sidebar session list + chat area.

Left panel: session list with highlighted active row, + button.
Right panel: active session's transcript + input row.
Sessions use pre-allocated containers (max 20) for NiceGUI compatibility.
"""

from __future__ import annotations

from typing import Any

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

MAX_SESSIONS = 20


# ── Session store (reactive, page-level) ──


class _ChatSession:
    """One conversation session."""
    __slots__ = ("container", "name", "orchestrator", "preview", "sid", "transcript")
    _counter = 0

    def __init__(self, name: str = "") -> None:
        _ChatSession._counter += 1
        self.sid = f"s{_ChatSession._counter}"
        self.name = name or f"Session {_ChatSession._counter}"
        self.orchestrator = AgentOrchestrator(settings=get_settings())
        self.transcript = ui.column().classes("w-full gap-2")
        self.preview = ""
        self.container = ui.column().classes("w-full overflow-auto flex-1 p-4")


# ── Helpers ──


def _fill_prompt(inp: ui.textarea, text: str) -> None:
    inp.value = text


def _trunc(s: str, n: int) -> str:
    return s[:n] + ("…" if len(s) > n else "")


async def _send(
    session: _ChatSession,
    message_input: ui.textarea,
    plan_card: ui.card,
    progress_card: ui.card,
    refresh_sidebar: Any,
) -> None:
    content = (message_input.value or "").strip()
    if not content:
        ui.notify("Enter a message first.", type="warning")
        return

    message_input.value = ""
    session.preview = _trunc(content, 60)
    refresh_sidebar()

    # ── User ──
    c = ui.card().classes("w-full bg-white border p-3")
    with c:
        ui.markdown(f"**🧑 You**  \n{content}")
    c.move(session.transcript)

    # ── Route ──
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

    # ── Orchestrate ──
    run_id = new_id("run")
    assistant_card: ui.card | None = None
    assistant_md: ui.markdown | None = None
    token_buf: list[str] = []
    lbl_ref: list[ui.label | None] = [None]
    need_new_card: bool = False

    try:
        async for evt in session.orchestrator.execute_stream(
            message=content, routing=decision, run_id=run_id,
        ):
            et = evt.type
            em = evt.message
            ed = evt.data

            if et == "run_started":
                exp = ed.get("experiment_id", "?")
                c2 = ui.card().classes("w-full bg-gray-50 p-2 text-xs text-gray-500")
                with c2:
                    ui.label(f"**Run** `{run_id[:8]}` | **Exp** `{exp}` | `{intent}`")
                c2.move(session.transcript)

            elif et == "progress":
                if em and lbl_ref[0] is not None:
                    lbl_ref[0].set_text(em)

            elif et == "token":
                if assistant_card is None or need_new_card:
                    assistant_card = ui.card().classes(
                        "w-full bg-blue-50 border p-3"
                    )
                    with assistant_card:
                        ui.markdown("**🤖 Assistant**").classes(
                            "text-sm font-semibold text-blue-800"
                        )
                        assistant_md = ui.markdown("").classes("text-sm text-blue-900")
                    assistant_card.move(session.transcript)
                    token_buf = []
                    need_new_card = False
                token_buf.append(em)
                if assistant_md is not None:
                    assistant_md.set_content("".join(token_buf))
                if len(token_buf) <= 8:
                    session.preview = _trunc("".join(token_buf), 60)
                    refresh_sidebar()

            elif et == "tool_start":
                tn = ed.get("tool_name", "")
                progress_card.clear()
                with progress_card:
                    with ui.row().classes("items-center gap-2"):
                        ui.spinner(size="sm")
                        lbl_ref[0] = ui.label(f"Executing: `{tn}`").classes("text-sm")
                progress_card.visible = True
                c2 = ui.card().classes("w-full bg-gray-50 border p-2 text-xs")
                with c2:
                    ui.markdown(f"🔧 **Calling:** `{tn}`")
                c2.move(session.transcript)
                need_new_card = True

            elif et == "tool_args":
                args = ed.get("arguments", {})
                import json as _json
                astr = _json.dumps(args, ensure_ascii=False, default=str)
                c2 = ui.card().classes("w-full bg-gray-50 border p-2 text-xs")
                with c2:
                    ui.markdown(f"```json\n{astr[:500]}\n```")
                c2.move(session.transcript)

            elif et == "tool_done":
                prv = ed.get("result_preview", "")
                progress_card.visible = False
                c2 = ui.card().classes("w-full bg-gray-50 border p-2 text-xs")
                with c2:
                    ui.markdown(f"✅ **Result:** `{prv}`")
                c2.move(session.transcript)

            elif et == "done":
                progress_card.visible = False
                plan_card.visible = False
                tc = ed.get("tool_calls_count", 0)
                c2 = ui.card().classes("w-full bg-green-50 border p-2")
                with c2:
                    ui.markdown(f"**✅ Done** — {tc} tool call(s).")
                c2.move(session.transcript)

            elif et == "error":
                progress_card.visible = False
                plan_card.visible = False
                c2 = ui.card().classes(
                    "w-full bg-red-50 border border-red-300 p-3"
                )
                with c2:
                    ui.icon("error", color="red").classes("inline")
                    ui.markdown(f"**❌ Error**  \n{em}")
                c2.move(session.transcript)

    except Exception as exc:
        progress_card.visible = False
        plan_card.visible = False
        c2 = ui.card().classes("w-full bg-red-50 border border-red-300 p-3")
        with c2:
            ui.markdown(f"**❌ Orchestration failed:** {exc}")
        c2.move(session.transcript)


# ── Page ──


def register() -> None:
    @ui.page("/")
    def chat_page() -> None:
        shell("Chat")

        # ── Session state
        _refresh_ref: list[Any] = [lambda: None]  # set after sidebar_list
        sessions: dict[str, _ChatSession] = {}
        active_sid: str = ""
        slot_containers: list[ui.column] = []  # pre-allocated

        # ═══ Create a session ═══
        def create_session(name: str = "", focus: bool = True) -> _ChatSession:
            if len(sessions) >= MAX_SESSIONS:
                ui.notify(f"Max {MAX_SESSIONS} sessions.", type="warning")
                raise RuntimeError("too many sessions")
            s = _ChatSession(name=name)
            sessions[s.sid] = s
            # Move transcript into its container
            s.transcript.move(s.container)
            # Assign a pre-allocated slot
            if slot_containers:
                slot = slot_containers.pop(0)
                s.container.move(slot)
            else:
                # Shouldn't happen with pre-allocation
                s.container.move(chat_stack)
            _refresh_ref[0]()
            if focus:
                activate_session(s.sid)
            return s

        def activate_session(sid: str) -> None:
            nonlocal active_sid
            if active_sid == sid:
                return
            # Hide current
            if active_sid and active_sid in sessions:
                sessions[active_sid].container.set_visibility(False)
            # Show new
            if sid in sessions:
                sessions[sid].container.set_visibility(True)
                active_sid = sid
            _refresh_ref[0]()

        def close_session(sid: str) -> None:
            nonlocal active_sid
            if len(sessions) <= 1:
                ui.notify("Cannot close the last session.", type="warning")
                return
            s = sessions.pop(sid, None)
            if s is not None:
                s.container.clear()
                s.container.set_visibility(False)
                slot_containers.append(s.container)
            if active_sid == sid and sessions:
                activate_session(next(iter(sessions)))

        async def send_handler() -> None:
            nonlocal active_sid
            if not active_sid or active_sid not in sessions:
                ui.notify("No active session.", type="warning")
                return
            s = sessions[active_sid]
            await _send(s, message, plan_card, progress_card, _refresh_ref[0])

        # ═══ Layout ═══
        with ui.row().classes("w-full gap-0 flex-1").style("min-height: 0"):
            # ── LEFT: session sidebar (220px) ──
            with ui.column().classes("border-r bg-gray-50/50").style(
                "width: 220px; flex-shrink: 0"
            ):
                with ui.row().classes(
                    "w-full items-center justify-between p-3 border-b"
                ):
                    ui.label("Sessions").classes(
                        "text-sm font-semibold text-gray-500 uppercase tracking-wide"
                    )
                    ui.button(
                        icon="add", on_click=lambda: create_session(),
                    ).props("flat round size=sm dense")

                # Refreshable session list
                @ui.refreshable
                def sidebar_list() -> None:
                    for sid, s in sessions.items():
                        is_active = sid == active_sid
                        bg = "bg-blue-50 border-l-[3px] border-l-blue-500" if is_active else ""
                        row = ui.row().classes(
                            f"w-full items-center gap-2 px-3 py-2.5 cursor-pointer "
                            f"hover:bg-gray-100 transition-colors {bg}"
                        )
                        with row:
                            ui.icon("chat", size="sm").classes(
                                "text-blue-500" if is_active else "text-gray-400"
                            )
                            with ui.column().classes("gap-0 flex-1 min-w-0"):
                                ui.label(s.name).classes(
                                    f"text-xs truncate "
                                    f"{'font-semibold text-blue-700' if is_active else ''}"
                                )
                                ui.label(s.preview or "还没有对话").classes(
                                    "text-[11px] text-gray-400 truncate"
                                )
                            if len(sessions) > 1:
                                ui.button(
                                    icon="close",
                                    on_click=lambda sid=sid: close_session(sid),
                                ).props("flat round size=xs dense").classes(
                                    "opacity-30 hover:opacity-100"
                                )
                        row.on("click", lambda sid=sid: activate_session(sid))

                sidebar_list()
                _refresh_ref[0] = sidebar_list.refresh

                # Footer
                with ui.row().classes("items-center gap-1 p-3 text-[11px] text-gray-400"):
                    st = get_settings()
                    if st.deepseek_api_key:
                        ui.icon("check_circle", size="xs", color="green")
                        ui.label(f"DeepSeek ({st.deepseek_model})")
                    else:
                        ui.icon("warning", size="xs", color="orange")
                        ui.label("No LLM")

            # ── RIGHT: chat area ──
            with ui.column().classes("flex-1 flex flex-col gap-0").style("min-width: 0"):
                # Stack of pre-allocated session containers
                chat_stack = ui.column().classes("w-full flex-1 overflow-auto relative")
                # Pre-allocate slots
                for _ in range(MAX_SESSIONS):
                    slot = ui.column().classes("w-full h-full absolute inset-0")
                    slot.set_visibility(False)
                    slot.move(chat_stack)
                    slot_containers.append(slot)

                # Plan + progress (shared)
                plan_card = ui.card().classes("w-full bg-blue-50 p-3 mx-4 mt-2")
                plan_card.visible = False

                progress_card = ui.card().classes("w-full bg-green-50 p-3 mx-4")
                with progress_card:
                    with ui.row().classes("items-center gap-2"):
                        ui.spinner(size="sm")
                        ui.label("").classes("text-sm")
                progress_card.visible = False

                # Input
                with ui.row().classes("w-full items-end gap-2 p-4 border-t"):
                    message = (
                        ui.textarea("Type your research question...")
                        .classes("grow")
                        .props("autogrow rows=2 outlined")
                    )
                    ui.button("Send", on_click=send_handler).props("color=primary")

                # Expandables
                with ui.expansion("Advanced", icon="tune").classes("w-full px-4"):
                    with ui.row().classes("gap-4"):
                        ui.select(
                            ["auto", "stock", "etf", "stock_etf"],
                            value="auto", label="Universe",
                        ).classes("w-40")
                        with ui.row().classes("gap-2"):
                            ui.input("Start", value="").classes("w-32").props("placeholder=auto")
                            ui.input("End", value="").classes("w-32").props("placeholder=auto")
                        ui.select(
                            ["balanced", "fast", "thorough"],
                            value="balanced", label="Budget",
                        ).classes("w-36")

                with ui.expansion(
                    "Suggested prompts", icon="lightbulb"
                ).classes("w-full px-4"):
                    with ui.row().classes("flex-wrap gap-2"):
                        for p in SUGGESTED_PROMPTS:
                            ui.chip(p, on_click=lambda _, text=p: _fill_prompt(message, text))

        # ── Initial session ──
        create_session(focus=True)
