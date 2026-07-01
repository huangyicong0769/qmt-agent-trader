"""Chat page — persistent multi-session conversations with interrupt control.

Interrupt levels (排队 / 引导 / 打断):
- 排队: Queue new message until current run finishes.
- 引导 (default): Append to conversation when current run completes.
- 打断: Cancel running execution, start fresh with new message.
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from nicegui import ui

from qmt_agent_trader.agent.orchestrator import AgentOrchestrator
from qmt_agent_trader.core.config import get_settings
from qmt_agent_trader.core.ids import new_id
from qmt_agent_trader.web.ui.layout import shell

SESSIONS_DIR = Path("sessions")
SESSIONS_DIR.mkdir(parents=True, exist_ok=True)

SUGGESTED_PROMPTS = [
    "帮我发现几个适合A股个股和ETF的低波动高胜率因子，并自动跑初步验证。",
    "列出当前数据湖中所有可用的因子，并验证 momentum_20d 因子。",
    "基于最近有效的候选因子，写一个日频轮动策略并回测。",
    "看看最近失败的实验，判断是不是缺少某个工具。",
    "解释一下上一个回测为什么收益高但回撤也大。",
]

INTERRUPT_LEVELS = {
    "guide": "引导",
    "queue": "排队",
    "interrupt": "打断",
}
INTERRUPT_LABELS = {v: k for k, v in INTERRUPT_LEVELS.items()}


# ── Persistent message model ──


@dataclass
class ChatMessage:
    role: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class _PendingMessage:
    level: str
    content: str
    ready_to_send: bool = False


@dataclass(frozen=True)
class _PendingMessageStatus:
    badge: str
    detail: str
    can_undo: bool
    can_send: bool


# ── Session model ──


class _ChatSession:
    __slots__ = (
        "_path",
        "container",
        "messages",
        "name",
        "orchestrator",
        "preview",
        "sid",
        "transcript",
    )
    _counter = 0

    def __init__(self, name: str = "", sid: str = "") -> None:
        if sid:
            self.sid = sid
        else:
            _ChatSession._counter += 1
            self.sid = f"s{_ChatSession._counter}"
        self.name = name or f"Session {_ChatSession._counter}"
        self.orchestrator = AgentOrchestrator(settings=get_settings())
        self.transcript = ui.column().classes("w-full gap-2")
        self.preview = ""
        self.container = ui.column().classes("w-full overflow-auto flex-1 p-4")
        self.messages: list[ChatMessage] = []
        self._path = SESSIONS_DIR / f"{self.sid}.json"

    def add_message(self, role: str, content: str, **meta: Any) -> None:
        msg = ChatMessage(role=role, content=content, metadata=dict(meta))
        self.messages.append(msg)
        self._update_preview(role, content)

    def _update_preview(self, role: str, content: str) -> None:
        if role == "user":
            self.preview = _trunc(content, 60)
        elif role == "assistant":
            self.preview = _trunc(content, 60)

    def save(self) -> None:
        data: dict[str, Any] = {
            "sid": self.sid,
            "name": self.name,
            "counter": _ChatSession._counter,
            "preview": self.preview,
            "messages": [
                {"role": m.role, "content": m.content, "metadata": m.metadata}
                for m in self.messages
            ],
        }
        self._path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def rebuild_ui(self) -> None:
        self.transcript.clear()
        for msg in self.messages:
            _render_stored_message(self.transcript, msg)

    @staticmethod
    def load_all() -> list[_ChatSession]:
        sessions: list[_ChatSession] = []
        max_counter = 0
        for fpath in sorted(
            SESSIONS_DIR.glob("s*.json"),
            key=lambda p: p.stat().st_mtime,
        ):
            try:
                data = json.loads(fpath.read_text(encoding="utf-8"))
            except Exception:
                continue
            sid = str(data.get("sid", fpath.stem))
            s = _ChatSession(
                name=str(data.get("name", "")),
                sid=sid,
            )
            s.preview = str(data.get("preview", ""))
            for mdata in data.get("messages", []):
                if isinstance(mdata, dict):
                    s.messages.append(ChatMessage(
                        role=str(mdata.get("role", "")),
                        content=str(mdata.get("content", "")),
                        metadata=mdata.get("metadata", {}) if isinstance(
                            mdata.get("metadata"), dict
                        ) else {},
                    ))
            sessions.append(s)
            try:
                n = int(sid.lstrip("s"))
                if n > max_counter:
                    max_counter = n
            except ValueError:
                pass
        _ChatSession._counter = max_counter
        return sessions


def _render_stored_message(transcript: ui.column, msg: ChatMessage) -> None:
    role = msg.role
    content = msg.content
    meta = msg.metadata

    if role == "user":
        c = ui.card().classes("w-full bg-white border p-3")
        with c:
            ui.markdown(f"**🧑 You**  \n{content}")
        c.move(transcript)

    elif role == "info":
        c = ui.card().classes("w-full bg-gray-50 p-2 text-xs text-gray-500")
        with c:
            ui.label(content)
        c.move(transcript)

    elif role == "assistant":
        c = ui.card().classes("w-full bg-blue-50 border p-3")
        with c:
            ui.markdown("**🤖 Assistant**").classes(
                "text-sm font-semibold text-blue-800"
            )
            ui.markdown(content).classes("text-sm text-blue-900")
        c.move(transcript)

    elif role == "tool":
        c = ui.card().classes("w-full bg-gray-50 border p-2 text-xs")
        tool_name = meta.get("tool_name", "")
        result = meta.get("result_preview", "")
        tool_phase = meta.get("phase", "done")
        if tool_phase == "start":
            with c:
                ui.markdown(f"🔧 **Calling:** `{tool_name}`")
        elif tool_phase == "args":
            with c:
                ui.markdown(f"```json\n{content[:500]}\n```")
        elif tool_phase == "done":
            with c:
                ui.markdown(f"✅ `{tool_name}`: {result}")
        c.move(transcript)

    elif role == "done":
        c = ui.card().classes("w-full bg-green-50 border p-2")
        with c:
            ui.markdown(f"**✅ Done** — {content}")
        c.move(transcript)

    elif role == "error":
        c = ui.card().classes("w-full bg-red-50 border border-red-300 p-3")
        with c:
            ui.icon("error", color="red").classes("inline")
            ui.markdown(f"**❌ Error**  \n{content}")
        c.move(transcript)


# ── Helpers ──


def _fill_prompt(inp: ui.textarea, text: str) -> None:
    inp.value = text


def _trunc(s: str, n: int) -> str:
    return s[:n] + ("…" if len(s) > n else "")


def _format_elapsed(total_seconds: int) -> str:
    seconds = max(0, int(total_seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:02d}:{seconds:02d}"


def _build_pending_message(level: str, raw_content: str) -> _PendingMessage | None:
    content = raw_content.strip()
    if not content:
        return None
    if level not in {"guide", "queue"}:
        level = "guide"
    return _PendingMessage(level=level, content=content)


def _mark_pending_ready(pending: _PendingMessage) -> _PendingMessage:
    return replace(pending, ready_to_send=True)


def _pending_message_status(
    pending: _PendingMessage,
    *,
    queue_depth: int | None = None,
) -> _PendingMessageStatus:
    label = INTERRUPT_LEVELS[pending.level]
    preview = _trunc(pending.content, 80)
    depth = max(1, queue_depth) if queue_depth is not None else None
    depth_suffix = f" · {depth}" if depth is not None else ""
    depth_detail = f"队列深度 {depth}。" if depth is not None else ""
    if pending.ready_to_send:
        return _PendingMessageStatus(
            badge=f"{label}待确认{depth_suffix}",
            detail=f"{depth_detail}{label}消息已就绪：{preview}。发送前可撤销。",
            can_undo=True,
            can_send=True,
        )
    detail_prefix = (
        "排队消息" if pending.level == "queue" else "引导消息"
    )
    return _PendingMessageStatus(
        badge=f"{label}中{depth_suffix}",
        detail=f"{depth_detail}{detail_prefix}等待当前任务完成：{preview}",
        can_undo=True,
        can_send=False,
    )


def _todo_status_markdown(data: dict[str, Any]) -> str:
    summary = data.get("summary", {}) if isinstance(data.get("summary"), dict) else {}
    items = data.get("items", []) if isinstance(data.get("items"), list) else []
    active = data.get("active_item")
    completed = summary.get("completed", 0)
    total = summary.get("total", len(items))
    lines = [f"**Todo** {completed}/{total} completed"]
    goal = data.get("goal")
    if goal:
        lines.append(f"Goal: {goal}")
    if isinstance(active, dict):
        lines.append(f"Active: {active.get('title', '')}")
    if not items:
        lines.append("No todo items.")
        return "\n\n".join(lines)
    lines.append("")
    for item in items:
        if not isinstance(item, dict):
            continue
        status = str(item.get("status", "PENDING"))
        title = str(item.get("title", ""))
        notes = str(item.get("notes", "")).strip()
        suffix = f" - {notes}" if notes else ""
        lines.append(f"- `{status}` {title}{suffix}")
    return "\n".join(lines)


async def _send(
    session: _ChatSession,
    message_input: ui.textarea,
    plan_card: ui.card,
    progress_card: ui.card,
    queue_badge: ui.badge,
    refresh_sidebar: Any,
    cancel_event: asyncio.Event,
) -> None:
    content = (message_input.value or "").strip()
    if not content:
        ui.notify("Enter a message first.", type="warning")
        return

    message_input.value = ""

    # ── User ──
    c = ui.card().classes("w-full bg-white border p-3")
    with c:
        ui.markdown(f"**🧑 You**  \n{content}")
    c.move(session.transcript)
    session.add_message("user", content)
    session.save()
    refresh_sidebar()

    # ── Orchestrate ──
    run_id = new_id("run")
    assistant_card: ui.card | None = None
    assistant_md: ui.markdown | None = None
    token_buf: list[str] = []
    final_text: str = ""
    lbl_ref: list[ui.label | None] = [None]
    need_new_card: bool = False
    cancelled = False

    try:
        async for evt in session.orchestrator.execute_stream(
            message=content,
            run_id=run_id,
            session_id=session.sid,
            history=[
                {"role": msg.role, "content": msg.content}
                for msg in session.messages
            ],
            cancel_event=cancel_event,
        ):
            # Check for cancellation
            if evt.type == "cancelled":
                cancelled = True
                break

            et = evt.type
            em = evt.message
            ed = evt.data

            if et == "run_started":
                exp = ed.get("experiment_id", "?")
                c2 = ui.card().classes("w-full bg-gray-50 p-2 text-xs text-gray-500")
                with c2:
                    ui.label(f"**Run** `{run_id[:8]}` | **Exp** `{exp}`")
                c2.move(session.transcript)
                session.add_message("info", f"Run {run_id[:8]}")

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
                        assistant_md = ui.markdown("").classes(
                            "text-sm text-blue-900"
                        )
                    assistant_card.move(session.transcript)
                    token_buf = []
                    need_new_card = False
                token_buf.append(em)
                if assistant_md is not None:
                    assistant_md.set_content("".join(token_buf))
                if len(token_buf) <= 8:
                    refresh_sidebar()

            elif et == "final_message":
                final_text = em
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
                    assistant_card.move(session.transcript)
                    need_new_card = False
                token_buf = [final_text]
                if assistant_md is not None:
                    assistant_md.set_content(final_text)

            elif et == "tool_start":
                tn = ed.get("tool_name", "")
                progress_card.clear()
                with progress_card:
                    with ui.row().classes("items-center gap-2"):
                        ui.spinner(size="sm")
                        lbl_ref[0] = ui.label(
                            f"Executing: `{tn}`"
                        ).classes("text-sm")
                progress_card.visible = True
                c2 = ui.card().classes("w-full bg-gray-50 border p-2 text-xs")
                with c2:
                    ui.markdown(f"🔧 **Calling:** `{tn}`")
                c2.move(session.transcript)
                session.add_message("tool", "", tool_name=tn, phase="start")
                need_new_card = True

            elif et == "tool_args":
                args = ed.get("arguments", {})
                astr = json.dumps(args, ensure_ascii=False, default=str)
                c2 = ui.card().classes("w-full bg-gray-50 border p-2 text-xs")
                with c2:
                    ui.markdown(f"```json\n{astr[:500]}\n```")
                c2.move(session.transcript)
                session.add_message(
                    "tool", astr,
                    tool_name=ed.get("tool_name", ""), phase="args",
                )

            elif et == "tool_done":
                prv = ed.get("result_preview", "")
                result_id = ed.get("result_id", "")
                progress_card.visible = False
                c2 = ui.card().classes("w-full bg-gray-50 border p-2 text-xs")
                with c2:
                    ui.markdown(f"✅ **Result:** `{prv}`")
                c2.move(session.transcript)
                session.add_message(
                    "tool", "",
                    tool_name=ed.get("tool_name", ""),
                    result_preview=prv, result_id=result_id, phase="done",
                )

            elif et == "todo_status":
                plan_card.clear()
                with plan_card:
                    ui.markdown(_todo_status_markdown(ed)).classes(
                        "text-sm text-blue-900"
                    )
                plan_card.visible = True

            elif et == "done":
                progress_card.visible = False
                plan_card.visible = False
                tc = ed.get("tool_calls_count", 0)
                c2 = ui.card().classes("w-full bg-green-50 border p-2")
                with c2:
                    ui.markdown(f"**✅ Done** — {tc} tool call(s).")
                c2.move(session.transcript)
                full_text = final_text or "".join(token_buf)
                if full_text:
                    session.add_message("assistant", full_text)
                session.add_message("done", f"{tc} tool call(s) completed.")
                session.save()
                refresh_sidebar()

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
                session.add_message("error", em)
                session.save()

    except Exception as exc:
        progress_card.visible = False
        plan_card.visible = False
        c2 = ui.card().classes("w-full bg-red-50 border border-red-300 p-3")
        with c2:
            ui.markdown(f"**❌ Orchestration failed:** {exc}")
        c2.move(session.transcript)
        session.add_message("error", str(exc))
        session.save()

    finally:
        if cancelled:
            progress_card.visible = False
            plan_card.visible = False
            c2 = ui.card().classes("w-full bg-yellow-50 border p-2")
            with c2:
                ui.markdown("**⏸️ Interrupted** — 上一个任务被打断。")
            c2.move(session.transcript)


# ── Page ──


def register() -> None:
    @ui.page("/")
    def chat_page() -> None:
        shell("Chat")

        loaded_sessions = _ChatSession.load_all()
        sessions: dict[str, _ChatSession] = {}
        active_sid: str = ""
        _refresh_fn: list[Any] = [lambda: None]

        # Interrupt state
        cancel_event = asyncio.Event()
        pending_messages: list[_PendingMessage] = []
        is_running: bool = False
        run_started_at: float | None = None
        pending_card: Any = None
        pending_label: Any = None
        pending_send_button: Any = None
        run_timer_badge: Any = None

        def activate_session(sid: str) -> None:
            nonlocal active_sid
            if active_sid == sid:
                return
            if active_sid and active_sid in sessions:
                sessions[active_sid].container.visible = False
            if sid in sessions:
                sessions[sid].container.visible = True
                active_sid = sid
            plan_card.visible = False
            progress_card.visible = False
            _refresh_fn[0]()

        def create_session(name: str = "", focus: bool = True) -> _ChatSession:
            s = _ChatSession(name=name)
            sessions[s.sid] = s
            s.transcript.move(s.container)
            s.container.move(chat_stack)
            s.container.visible = False
            s.save()
            _refresh_fn[0]()
            if focus:
                activate_session(s.sid)
            return s

        def close_session(sid: str) -> None:
            nonlocal active_sid
            if len(sessions) <= 1:
                ui.notify("Cannot close the last session.", type="warning")
                return
            s = sessions.pop(sid, None)
            if s is not None:
                try:
                    s._path.unlink(missing_ok=True)
                except Exception:
                    pass
                s.container.clear()
                s.container.delete()
            if active_sid == sid:
                first = next(iter(sessions))
                _refresh_fn[0]()
                activate_session(first)
            else:
                _refresh_fn[0]()

        def render_pending_status() -> None:
            if pending_card is None or pending_label is None:
                return
            if not pending_messages:
                queue_badge.visible = False
                pending_card.visible = False
                if pending_send_button is not None:
                    pending_send_button.visible = False
                queue_badge.update()
                pending_card.update()
                return

            status = _pending_message_status(
                pending_messages[0],
                queue_depth=len(pending_messages),
            )
            queue_badge.set_text(status.badge)
            queue_badge.visible = True
            pending_label.set_text(status.detail)
            pending_card.visible = True
            if pending_send_button is not None:
                pending_send_button.visible = status.can_send and not is_running
                pending_send_button.update()
            queue_badge.update()
            pending_card.update()

        def render_run_timer() -> None:
            if run_timer_badge is None:
                return
            if not is_running or run_started_at is None:
                run_timer_badge.visible = False
                run_timer_badge.update()
                return
            elapsed = int(time.monotonic() - run_started_at)
            run_timer_badge.set_text(f"运行 {_format_elapsed(elapsed)}")
            run_timer_badge.visible = True
            run_timer_badge.update()

        def cancel_pending_message() -> None:
            if not pending_messages:
                return
            pending_messages.pop(0)
            render_pending_status()
            ui.notify("已撤销当前待发送消息。", type="info")

        async def send_pending_message() -> None:
            if not pending_messages:
                return
            if is_running:
                ui.notify("当前任务仍在运行，稍后再确认发送。", type="warning")
                return
            content = pending_messages.pop(0).content
            render_pending_status()
            message.value = content
            await run_send_handler(from_pending=True)

        async def send_handler() -> None:
            await run_send_handler(from_pending=False)

        async def run_send_handler(*, from_pending: bool) -> None:
            nonlocal active_sid, is_running, run_started_at
            if not active_sid or active_sid not in sessions:
                ui.notify("No active session.", type="warning")
                return

            if pending_messages and pending_messages[0].ready_to_send and not from_pending:
                ui.notify("请先发送或撤销待处理消息。", type="warning")
                return

            level = INTERRUPT_LABELS.get(interrupt_select.value, "guide")

            if is_running:
                if level == "interrupt":
                    # 打断: cancel current, run new
                    cancel_event.set()
                    cancel_event.clear()
                    # Wait briefly for cancellation to propagate
                    await asyncio.sleep(0.1)
                    is_running = False
                    render_pending_status()
                    # Fall through to start new run
                elif level == "queue":
                    pending = _build_pending_message(level, message.value or "")
                    if pending is None:
                        ui.notify("Enter a message first.", type="warning")
                        return
                    pending_messages.append(pending)
                    message.value = ""
                    render_pending_status()
                    ui.notify("已排队，当前任务完成后等待你确认发送。", type="info")
                    return
                else:
                    pending = _build_pending_message(level, message.value or "")
                    if pending is None:
                        ui.notify("Enter a message first.", type="warning")
                        return
                    pending_messages.append(pending)
                    message.value = ""
                    render_pending_status()
                    ui.notify("已加入引导，当前任务完成后等待你确认发送。", type="info")
                    return

            is_running = True
            run_started_at = time.monotonic()
            render_pending_status()
            render_run_timer()

            try:
                await _send(
                    sessions[active_sid],
                    message,
                    plan_card,
                    progress_card,
                    queue_badge,
                    _refresh_fn[0],
                    cancel_event,
                )
            finally:
                is_running = False
                run_started_at = None
                render_run_timer()
                if pending_messages:
                    pending_messages[:] = [
                        _mark_pending_ready(pending) for pending in pending_messages
                    ]
                    render_pending_status()
                    ui.notify(
                        f"{len(pending_messages)} 条待发送消息已就绪，请确认发送或撤销。",
                        type="info",
                    )
                else:
                    render_pending_status()

        # ═══ Layout ═══
        with ui.row().classes("w-full gap-0 flex-1").style("min-height: 0"):
            # ── LEFT: session sidebar ──
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

                @ui.refreshable
                def sidebar_list() -> None:
                    for sid, s in sessions.items():
                        is_active = sid == active_sid
                        bg = (
                            "bg-blue-50 border-l-[3px] border-l-blue-500"
                            if is_active else ""
                        )
                        row = ui.row().classes(
                            f"w-full items-center gap-2 px-3 py-2.5 "
                            f"cursor-pointer hover:bg-gray-100 "
                            f"transition-colors {bg}"
                        )
                        with row:
                            ui.icon("chat", size="sm").classes(
                                "text-blue-500" if is_active else "text-gray-400"
                            )
                            with ui.column().classes("gap-0 flex-1 min-w-0"):
                                nc = (
                                    "font-semibold text-blue-700"
                                    if is_active else ""
                                )
                                ui.label(s.name).classes(f"text-xs truncate {nc}")
                                ui.label(
                                    s.preview or "还没有对话"
                                ).classes("text-[11px] text-gray-400 truncate")
                            if len(sessions) > 1:
                                ui.button(
                                    icon="close",
                                    on_click=lambda sid=sid: close_session(sid),
                                ).props("flat round size=xs dense").classes(
                                    "opacity-30 hover:opacity-100"
                                )
                        row.on("click", lambda sid=sid: activate_session(sid))

                sidebar_list()
                _refresh_fn[0] = sidebar_list.refresh

                with ui.row().classes(
                    "items-center gap-1 p-3 text-[11px] text-gray-400"
                ):
                    st = get_settings()
                    if st.deepseek_api_key:
                        ui.icon("check_circle", size="xs", color="green")
                        ui.label(f"DeepSeek ({st.deepseek_model})")
                    else:
                        ui.icon("warning", size="xs", color="orange")
                        ui.label("No LLM")

            # ── RIGHT: chat area ──
            with ui.column().classes("flex-1 flex flex-col gap-0").style(
                "min-width: 0"
            ):
                chat_stack = ui.column().classes(
                    "w-full flex-1 overflow-auto relative"
                )

                plan_card = ui.card().classes("w-full bg-blue-50 p-3 mx-4 mt-2")
                plan_card.visible = False

                progress_card = ui.card().classes("w-full bg-green-50 p-3 mx-4")
                with progress_card:
                    with ui.row().classes("items-center gap-2"):
                        ui.spinner(size="sm")
                        ui.label("").classes("text-sm")
                progress_card.visible = False

                pending_card = ui.card().classes(
                    "w-full bg-amber-50 border border-amber-200 p-3 mx-4"
                )
                with pending_card:
                    with ui.row().classes("w-full items-center gap-2"):
                        ui.icon("schedule", size="sm").classes("text-amber-700")
                        pending_label = ui.label("").classes(
                            "text-sm text-amber-900 grow"
                        )
                        ui.button(
                            "撤销",
                            icon="undo",
                            on_click=cancel_pending_message,
                        ).props("flat dense color=negative")
                        pending_send_button = ui.button(
                            "发送",
                            icon="send",
                            on_click=send_pending_message,
                        ).props("dense color=primary")
                pending_card.visible = False
                pending_send_button.visible = False

                # ── Input row with interrupt control ──
                with ui.row().classes("w-full items-end gap-2 p-4 border-t"):
                    message = (
                        ui.textarea("Type your research question...")
                        .classes("grow")
                        .props("autogrow rows=2 outlined")
                    )
                    # Interrupt level selector
                    interrupt_select = ui.select(
                        list(INTERRUPT_LABELS.keys()),
                        label="等级",
                        value="引导",
                    ).classes("w-20").props("dense outlined")

                    run_timer_badge = ui.badge("", color="green")
                    run_timer_badge.visible = False

                    queue_badge = ui.badge("", color="orange")
                    queue_badge.visible = False

                    ui.button("Send", on_click=send_handler).props("color=primary")

                ui.timer(1.0, render_run_timer)

                with ui.expansion("Advanced", icon="tune").classes("w-full px-4"):
                    with ui.row().classes("gap-4"):
                        ui.select(
                            ["auto", "stock", "etf", "stock_etf"],
                            value="auto", label="Universe",
                        ).classes("w-40")
                        with ui.row().classes("gap-2"):
                            ui.input("Start", value="").classes("w-32").props(
                                "placeholder=auto"
                            )
                            ui.input("End", value="").classes("w-32").props(
                                "placeholder=auto"
                            )
                        ui.select(
                            ["balanced", "fast", "thorough"],
                            value="balanced", label="Budget",
                        ).classes("w-36")

                with ui.expansion(
                    "Suggested prompts", icon="lightbulb"
                ).classes("w-full px-4"):
                    with ui.row().classes("flex-wrap gap-2"):
                        for p in SUGGESTED_PROMPTS:
                            ui.chip(
                                p,
                                on_click=lambda _, text=p: _fill_prompt(
                                    message, text
                                ),
                            )

        # ── Load existing sessions ──
        for s in loaded_sessions:
            sessions[s.sid] = s
            s.transcript.move(s.container)
            s.container.move(chat_stack)
            s.container.visible = False
            s.rebuild_ui()

        if sessions:
            _refresh_fn[0]()
            activate_session(next(iter(sessions)))
        else:
            create_session(focus=True)
