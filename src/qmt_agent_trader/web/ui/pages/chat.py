"""Chat page: a disposable NiceGUI subscriber for application-owned Runs."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import Any, Literal

from nicegui import Client, ui

from qmt_agent_trader.core.config import get_settings
from qmt_agent_trader.web.chat_repository import (
    ChatSessionRepository,
    build_chat_session_repository,
)
from qmt_agent_trader.web.chat_run_manager import (
    ChatRunManager,
    RunAlreadyActiveError,
    RunEvent,
    RunSnapshot,
    RunStatus,
    SessionDeletionBlockedError,
    SuccessorAlreadyPendingError,
    is_terminal_run_event,
)
from qmt_agent_trader.web.schemas import ChatMessage as StoredChatMessage
from qmt_agent_trader.web.schemas import ChatSession as StoredChatSession
from qmt_agent_trader.web.ui.layout import shell

logger = logging.getLogger(__name__)

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
INTERRUPT_LABELS = {value: key for key, value in INTERRUPT_LEVELS.items()}
SIDEBAR_REFRESH_EVENT_TYPES = frozenset(
    {"user_message", "final_message", "error", "done", "cancelled"}
)


@dataclass
class ChatMessage:
    role: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    stored: StoredChatMessage | None = None


@dataclass(frozen=True)
class _PendingMessage:
    level: str
    content: str
    ready_to_send: bool = False
    session_id: str = ""


@dataclass(frozen=True)
class _PendingMessageStatus:
    badge: str
    detail: str
    can_undo: bool
    can_send: bool


@dataclass
class _AssistantRenderState:
    text: str = ""
    card: Any = None
    markdown: Any = None
    persisted_final: bool = False
    flush_task: asyncio.Task[Any] | None = None


@dataclass
class _RunRenderTargets:
    transcript: Any
    plan_card: Any = None
    progress_card: Any = None
    progress_label: Any = None


class _ChatSession:
    __slots__ = (
        "_initial_preview",
        "_repository",
        "_stored",
        "container",
        "messages",
        "name",
        "preview",
        "sid",
        "transcript",
    )
    _counter = 0

    def __init__(
        self,
        name: str = "",
        sid: str = "",
        *,
        repository: ChatSessionRepository | None = None,
    ) -> None:
        if sid:
            self.sid = sid
        else:
            _ChatSession._counter += 1
            self.sid = f"s{_ChatSession._counter}"
        self.name = name or f"Session {_ChatSession._counter}"
        self.transcript = ui.column().classes("w-full gap-2")
        self.preview = ""
        self.container = ui.column().classes("w-full overflow-auto flex-1 p-4")
        self.messages: list[ChatMessage] = []
        self._repository = repository or _chat_repository()
        self._stored: StoredChatSession | None = None
        self._initial_preview = ""

    def add_message(self, role: str, content: str, **meta: Any) -> None:
        message = ChatMessage(role=role, content=content, metadata=dict(meta))
        self.messages.append(message)
        self._update_preview(role, content)

    def _update_preview(self, role: str, content: str) -> None:
        if role in {"user", "assistant"}:
            self.preview = _trunc(content, 60)

    def save(self) -> None:
        messages = [
            (
                message.stored.model_copy(
                    update={
                        "role": message.role,
                        "content": message.content,
                        "metadata": message.metadata,
                    }
                )
                if message.stored is not None
                else StoredChatMessage(
                    session_id=self.sid,
                    role=message.role,
                    content=message.content,
                    metadata=message.metadata,
                )
            )
            for message in self.messages
        ]
        if self._stored is None:
            legacy_ui = {"counter": _ChatSession._counter, "preview": self.preview}
            candidate = StoredChatSession(
                session_id=self.sid,
                title=self.name,
                messages=messages,
                context={"legacy_ui": legacy_ui},
            )
            expected_revision = 0
        else:
            context = dict(self._stored.context)
            if self.preview != self._initial_preview:
                current_ui = context.get("legacy_ui", {})
                legacy_ui = dict(current_ui) if isinstance(current_ui, dict) else {}
                legacy_ui["preview"] = self.preview
                context["legacy_ui"] = legacy_ui
            candidate = self._stored.model_copy(
                update={"title": self.name, "messages": messages, "context": context}
            )
            expected_revision = self._stored.revision
            if candidate == self._stored:
                return
        self._stored = self._repository.save(candidate, expected_revision=expected_revision)
        self._initial_preview = self.preview
        for ui_message, stored_message in zip(self.messages, self._stored.messages, strict=True):
            ui_message.stored = stored_message

    def rebuild_ui(
        self,
        *,
        client: Client | None = None,
        page_alive: bool | Callable[[], bool] = True,
    ) -> None:
        if client is not None and not _ui_target_alive(client, self.transcript, page_alive):
            return
        self.transcript.clear()
        for message in self.messages:
            _render_stored_message(
                self.transcript,
                message,
                client=client,
                page_alive=page_alive,
            )

    @staticmethod
    def load_all(repository: ChatSessionRepository | None = None) -> list[_ChatSession]:
        repository = repository or _chat_repository()
        sessions: list[_ChatSession] = []
        max_counter = 0
        for stored in reversed(repository.list()):
            session = _ChatSession(
                name=stored.title,
                sid=stored.session_id,
                repository=repository,
            )
            session._stored = stored
            legacy = stored.context.get("legacy_ui", {})
            session.preview = str(legacy.get("preview", "")) if isinstance(legacy, dict) else ""
            session._initial_preview = session.preview
            if isinstance(legacy, dict):
                try:
                    max_counter = max(max_counter, int(legacy.get("counter", 0)))
                except (TypeError, ValueError):
                    logger.debug("invalid legacy session counter for %s", stored.session_id)
            for message in stored.messages:
                session.messages.append(
                    ChatMessage(
                        role=message.role,
                        content=message.content,
                        metadata=message.metadata,
                        stored=message,
                    )
                )
            sessions.append(session)
            try:
                max_counter = max(max_counter, int(stored.session_id.lstrip("s")))
            except ValueError:
                pass
        _ChatSession._counter = max_counter
        return sessions


def _chat_repository() -> ChatSessionRepository:
    return build_chat_session_repository(get_settings())


def _page_alive_value(page_alive: bool | Callable[[], bool]) -> bool:
    return page_alive() if callable(page_alive) else page_alive


def _ui_target_alive(
    client: Client | None,
    target: Any,
    page_alive: bool | Callable[[], bool],
) -> bool:
    if not _page_alive_value(page_alive):
        return False
    if client is not None and client.is_deleted:
        return False
    return target is not None and not getattr(target, "is_deleted", False)


def _should_refresh_sidebar(event: RunEvent) -> bool:
    return event.event_type in SIDEBAR_REFRESH_EVENT_TYPES


def _can_start_successor_subscription(
    successor_session_id: str,
    active_session_id: str,
    session_ids: set[str],
    *,
    page_alive: bool,
) -> bool:
    return (
        page_alive
        and successor_session_id == active_session_id
        and successor_session_id in session_ids
    )


def _pending_messages_for(
    pending_messages_by_session: dict[str, list[_PendingMessage]],
    session_id: str,
) -> list[_PendingMessage]:
    return pending_messages_by_session.setdefault(session_id, [])


def _pending_message_session(pending: _PendingMessage, fallback_session_id: str) -> str:
    return pending.session_id or fallback_session_id


def _render_stored_message(
    transcript: Any,
    message: ChatMessage,
    *,
    client: Client | None = None,
    page_alive: bool | Callable[[], bool] = True,
) -> None:
    if client is not None and not _ui_target_alive(client, transcript, page_alive):
        return
    if client is None:
        _render_stored_message_in_context(transcript, message)
    else:
        with client:
            _render_stored_message_in_context(transcript, message)


def _render_stored_message_in_context(transcript: Any, message: ChatMessage) -> None:
    role = message.role
    content = message.content
    meta = message.metadata
    with transcript:
        if role == "user":
            with ui.card().classes("w-full bg-white border p-3"):
                ui.markdown(f"**🧑 You**  \n{content}")
        elif role == "info":
            with ui.card().classes("w-full bg-gray-50 p-2 text-xs text-gray-500"):
                ui.label(content)
        elif role == "assistant":
            with ui.card().classes("w-full bg-blue-50 border p-3"):
                ui.markdown("**🤖 Assistant**").classes("text-sm font-semibold text-blue-800")
                ui.markdown(content).classes("text-sm text-blue-900")
        elif role == "tool":
            tool_name = meta.get("tool_name", "")
            result = meta.get("result_preview", "")
            phase = meta.get("phase", "done")
            with ui.card().classes("w-full bg-gray-50 border p-2 text-xs"):
                if phase == "start":
                    ui.markdown(f"🔧 **Calling:** `{tool_name}`")
                elif phase == "args":
                    ui.markdown(f"```json\n{content[:500]}\n```")
                elif phase == "done":
                    ui.markdown(f"✅ `{tool_name}`: {result}")
        elif role == "done":
            with ui.card().classes("w-full bg-green-50 border p-2"):
                ui.markdown(f"**✅ Done** — {content}")
        elif role == "error":
            with ui.card().classes("w-full bg-red-50 border border-red-300 p-3"):
                ui.icon("error", color="red").classes("inline")
                ui.markdown(f"**❌ Error**  \n{content}")


def _render_user_message(
    client: Client,
    transcript: Any,
    content: str,
    *,
    page_alive: bool | Callable[[], bool],
) -> None:
    if not _ui_target_alive(client, transcript, page_alive):
        return
    with client:
        with transcript:
            with ui.card().classes("w-full bg-white border p-3"):
                ui.markdown(f"**🧑 You**  \n{content}")


def _render_assistant_draft(
    client: Client,
    transcript: Any,
    state: _AssistantRenderState,
    *,
    page_alive: bool | Callable[[], bool],
) -> None:
    if state.markdown is not None and not getattr(state.markdown, "is_deleted", False):
        if _ui_target_alive(client, transcript, page_alive):
            state.markdown.set_content(state.text)
        return
    if not _ui_target_alive(client, transcript, page_alive):
        return
    with client:
        with transcript:
            with ui.card().classes("w-full bg-blue-50 border p-3") as card:
                ui.markdown("**🤖 Assistant**").classes(
                    "text-sm font-semibold text-blue-800"
                )
                markdown = ui.markdown(state.text).classes("text-sm text-blue-900")
    state.card = card
    state.markdown = markdown


def _render_info_event(
    client: Client,
    transcript: Any,
    content: str,
    *,
    page_alive: bool | Callable[[], bool],
    color: str = "gray",
) -> None:
    if not _ui_target_alive(client, transcript, page_alive):
        return
    with client:
        with transcript:
            with ui.card().classes(f"w-full bg-{color}-50 p-2 text-xs text-gray-500"):
                ui.label(content)


def _render_tool_event(
    client: Client,
    transcript: Any,
    event: RunEvent,
    *,
    page_alive: bool | Callable[[], bool],
) -> None:
    if not _ui_target_alive(client, transcript, page_alive):
        return
    data = event.data
    with client:
        with transcript:
            with ui.card().classes("w-full bg-gray-50 border p-2 text-xs"):
                if event.event_type == "tool_start":
                    ui.markdown(f"🔧 **Calling:** `{data.get('tool_name', '')}`")
                elif event.event_type == "tool_args":
                    arguments = json.dumps(
                        data.get("arguments", {}), ensure_ascii=False, default=str
                    )
                    ui.markdown(f"```json\n{arguments[:500]}\n```")
                else:
                    ui.markdown(f"✅ **Result:** `{data.get('result_preview', '')}`")


def _set_card_visible(
    client: Client,
    card: Any,
    visible: bool,
    *,
    page_alive: bool | Callable[[], bool],
) -> None:
    if not _ui_target_alive(client, card, page_alive):
        return
    card.visible = visible
    card.update()


def _render_run_snapshot(
    client: Client,
    targets: _RunRenderTargets,
    snapshot: dict[str, Any],
    state: _AssistantRenderState,
    *,
    page_alive: bool | Callable[[], bool],
) -> None:
    if not _ui_target_alive(client, targets.transcript, page_alive):
        return
    status = str(snapshot.get("status", ""))
    draft = str(snapshot.get("accumulated_draft", ""))
    if (
        draft
        and not state.persisted_final
        and status not in {RunStatus.COMPLETED.value, RunStatus.CANCELLED.value}
    ):
        state.text = draft
        _render_assistant_draft(
            client,
            targets.transcript,
            state,
            page_alive=page_alive,
        )


def _render_run_event(
    client: Client,
    targets: _RunRenderTargets,
    event: RunEvent,
    state: _AssistantRenderState,
    *,
    page_alive: bool | Callable[[], bool],
    defer_draft: bool = False,
) -> None:
    if not _ui_target_alive(client, targets.transcript, page_alive):
        return
    event_type = event.event_type
    if event_type == "snapshot":
        snapshot = event.data.get("snapshot", {})
        if isinstance(snapshot, dict):
            _render_run_snapshot(
                client,
                targets,
                snapshot,
                state,
                page_alive=page_alive,
            )
    elif event_type == "user_message":
        _render_user_message(
            client,
            targets.transcript,
            event.message,
            page_alive=page_alive,
        )
    elif event_type == "run_started":
        _render_info_event(
            client,
            targets.transcript,
            event.message,
            page_alive=page_alive,
        )
    elif event_type == "token":
        state.text += event.message
        if not defer_draft:
            _render_assistant_draft(
                client,
                targets.transcript,
                state,
                page_alive=page_alive,
            )
    elif event_type == "final_message":
        state.text = event.message or str(event.data.get("content", ""))
        _render_assistant_draft(
            client,
            targets.transcript,
            state,
            page_alive=page_alive,
        )
        state.persisted_final = True
    elif event_type in {"tool_start", "tool_args", "tool_done"}:
        if event_type == "tool_start":
            if targets.progress_card is not None:
                if _ui_target_alive(client, targets.progress_card, page_alive):
                    with client:
                        targets.progress_card.clear()
                        with targets.progress_card:
                            with ui.row().classes("items-center gap-2"):
                                ui.spinner(size="sm")
                                targets.progress_label = ui.label(
                                    f"Executing: `{event.data.get('tool_name', '')}`"
                                ).classes("text-sm")
                _set_card_visible(
                    client,
                    targets.progress_card,
                    True,
                    page_alive=page_alive,
                )
        elif event_type == "tool_done" and targets.progress_card is not None:
            _set_card_visible(
                client,
                targets.progress_card,
                False,
                page_alive=page_alive,
            )
        _render_tool_event(
            client,
            targets.transcript,
            event,
            page_alive=page_alive,
        )
    elif event_type == "progress":
        if (
            targets.progress_label is not None
            and not getattr(targets.progress_label, "is_deleted", False)
            and _page_alive_value(page_alive)
        ):
            targets.progress_label.set_text(event.message)
    elif event_type == "todo_status":
        if targets.plan_card is not None and _ui_target_alive(
            client, targets.plan_card, page_alive
        ):
            with client:
                targets.plan_card.clear()
                with targets.plan_card:
                    ui.markdown(_todo_status_markdown(event.data)).classes(
                        "text-sm text-blue-900"
                    )
            _set_card_visible(client, targets.plan_card, True, page_alive=page_alive)
    elif event_type == "cancelling":
        _render_info_event(
            client,
            targets.transcript,
            event.message,
            page_alive=page_alive,
            color="yellow",
        )
    elif event_type == "cancelled":
        _set_card_visible(client, targets.progress_card, False, page_alive=page_alive)
        _set_card_visible(client, targets.plan_card, False, page_alive=page_alive)
        _render_info_event(
            client,
            targets.transcript,
            f"⏸️ {event.message or 'Run cancelled.'}",
            page_alive=page_alive,
            color="yellow",
        )
    elif event_type == "done":
        _set_card_visible(client, targets.progress_card, False, page_alive=page_alive)
        _set_card_visible(client, targets.plan_card, False, page_alive=page_alive)
        tool_calls = event.data.get("tool_calls_count", 0)
        with client:
            with targets.transcript:
                with ui.card().classes("w-full bg-green-50 border p-2"):
                    ui.markdown(f"**✅ Done** — {tool_calls} tool call(s).")
    elif event_type == "error":
        _set_card_visible(client, targets.progress_card, False, page_alive=page_alive)
        _set_card_visible(client, targets.plan_card, False, page_alive=page_alive)
        with client:
            with targets.transcript:
                with ui.card().classes("w-full bg-red-50 border border-red-300 p-3"):
                    ui.icon("error", color="red").classes("inline")
                    ui.markdown(f"**❌ Error**  \n{event.message}")


def _fill_prompt(inp: Any, text: str) -> None:
    inp.value = text


def _trunc(value: str, length: int) -> str:
    return value[:length] + ("…" if len(value) > length else "")


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
    detail_prefix = "排队消息" if pending.level == "queue" else "引导消息"
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


def _event_sequence_for_session(session: _ChatSession, run_id: str) -> int:
    sequences = [
        int(message.metadata["event_sequence"])
        for message in session.messages
        if message.metadata.get("run_id") == run_id
        and isinstance(message.metadata.get("event_sequence"), int)
    ]
    return max(sequences, default=0)


def _append_local_event(session: _ChatSession, event: RunEvent) -> None:
    if event.sequence <= 0 or event.event_type in {"snapshot", "token", "progress"}:
        return
    marker = {
        "run_id": event.run_id,
        "event_sequence": event.sequence,
        "event_type": event.event_type,
    }
    if any(
        all(message.metadata.get(key) == value for key, value in marker.items())
        for message in session.messages
    ):
        return
    if event.event_type == "user_message":
        session.add_message("user", event.message, **marker)
    elif event.event_type == "final_message":
        session.add_message("assistant", event.message, **marker)
    elif event.event_type == "done":
        session.add_message("done", event.message, **marker)
    elif event.event_type == "error":
        session.add_message("error", event.message, **marker)
    elif event.event_type in {"run_started", "cancelling", "cancelled"}:
        session.add_message("info", event.message, **marker)
    elif event.event_type in {"tool_start", "tool_args", "tool_done"}:
        data = event.data
        metadata = {
            **marker,
            "tool_name": data.get("tool_name", ""),
            "phase": event.event_type.removeprefix("tool_"),
        }
        content = (
            json.dumps(data.get("arguments", {}), ensure_ascii=False, default=str)
            if event.event_type == "tool_args"
            else ""
        )
        if event.event_type == "tool_done":
            metadata.update(
                {
                    "result_id": data.get("result_id", ""),
                    "result_preview": data.get("result_preview", ""),
                }
            )
        session.add_message("tool", content, **metadata)


def register() -> None:
    @ui.page("/")
    def chat_page(client: Client) -> None:
        shell("Chat")
        repository = _chat_repository()
        manager: ChatRunManager
        from qmt_agent_trader.web.runtime import get_chat_run_manager

        manager = get_chat_run_manager()
        loaded_sessions = _ChatSession.load_all(repository)
        page_alive = True
        subscription_task: asyncio.Task[Any] | None = None
        page_tasks: set[asyncio.Task[Any]] = set()
        successor_watchers: dict[str, asyncio.Task[Any]] = {}
        subscribed_run_id: str | None = None
        active_sid = ""
        pending_messages_by_session: dict[str, list[_PendingMessage]] = {}
        last_rendered_sequence: dict[str, int] = {}
        assistant_states: dict[str, _AssistantRenderState] = {}
        run_started_at: dict[str, float] = {}
        _refresh_fn: list[Any] = [lambda: None]

        def alive() -> bool:
            return page_alive and not client.is_deleted

        def track_page_task(task: asyncio.Task[Any]) -> asyncio.Task[Any]:
            page_tasks.add(task)
            task.add_done_callback(page_tasks.discard)
            return task

        def safe_notify(
            message: str,
            *,
            type: Literal["positive", "negative", "warning", "info", "ongoing"] = "info",
        ) -> None:
            if not alive():
                return
            try:
                with client:
                    ui.notify(message, type=type)
            except Exception:
                logger.exception("chat page notification failed")

        def refresh_sidebar_safe() -> None:
            if not alive():
                return
            try:
                with client:
                    _refresh_fn[0]()
            except Exception:
                logger.exception("chat sidebar refresh failed")

        def active_run(sid: str | None = None) -> RunSnapshot | None:
            return manager.get_active_run(sid or active_sid) if sid or active_sid else None

        def run_is_busy(sid: str | None = None) -> bool:
            return active_run(sid) is not None

        def pending_messages_for(sid: str | None = None) -> list[_PendingMessage]:
            target_sid = sid or active_sid
            if not target_sid:
                return []
            return _pending_messages_for(pending_messages_by_session, target_sid)

        def current_pending_messages() -> list[_PendingMessage]:
            return pending_messages_for(active_sid)

        def cancel_current_subscription() -> None:
            nonlocal subscription_task, subscribed_run_id
            if subscription_task is not None and not subscription_task.done():
                subscription_task.cancel()
            subscription_task = None
            subscribed_run_id = None

        async def watch_successor(sid: str, old_run_id: str) -> None:
            try:
                while alive():
                    current = manager.get_active_run(sid)
                    if current is not None and current.run_id != old_run_id:
                        if _can_start_successor_subscription(
                            sid,
                            active_sid,
                            set(sessions),
                            page_alive=page_alive,
                        ):
                            start_subscription(sessions[sid], current)
                        return
                    if not manager.has_pending_successor(sid) and current is None:
                        return
                    await asyncio.sleep(0.05)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("successor watcher failed for session %s", sid)

        def schedule_successor_watch(sid: str, old_run_id: str) -> None:
            existing = successor_watchers.get(sid)
            if existing is not None and not existing.done():
                return
            task = track_page_task(asyncio.create_task(watch_successor(sid, old_run_id)))
            successor_watchers[sid] = task

        async def consume_run_events(session: _ChatSession, snapshot: RunSnapshot) -> None:
            run_id = snapshot.run_id
            cursor = last_rendered_sequence.get(
                run_id,
                _event_sequence_for_session(session, run_id),
            )
            last_rendered_sequence[run_id] = cursor
            state = assistant_states.setdefault(run_id, _AssistantRenderState())
            state.persisted_final = any(
                message.metadata.get("run_id") == run_id
                and message.metadata.get("event_type") == "final_message"
                for message in session.messages
            )
            targets = _RunRenderTargets(
                transcript=session.transcript,
                plan_card=plan_card,
                progress_card=progress_card,
            )

            def flush_draft_now() -> None:
                task = state.flush_task
                if task is not None and not task.done():
                    task.cancel()
                state.flush_task = None
                if alive():
                    _render_assistant_draft(
                        client,
                        targets.transcript,
                        state,
                        page_alive=alive,
                    )

            def schedule_draft_flush() -> None:
                if not alive():
                    return
                if state.flush_task is not None and not state.flush_task.done():
                    return

                async def flush_after_batch() -> None:
                    try:
                        await asyncio.sleep(0.05)
                        if alive():
                            _render_assistant_draft(
                                client,
                                targets.transcript,
                                state,
                                page_alive=alive,
                            )
                    except asyncio.CancelledError:
                        raise
                    finally:
                        if state.flush_task is asyncio.current_task():
                            state.flush_task = None

                state.flush_task = track_page_task(
                    asyncio.create_task(
                        flush_after_batch(),
                        name=f"chat-draft-flush-{run_id}",
                    )
                )

            subscription = manager.subscribe(run_id, after_sequence=cursor)
            try:
                async for event in subscription:
                    if not alive():
                        return
                    if event.sequence > 0 and event.sequence <= cursor:
                        continue
                    if event.event_type == "token":
                        _render_run_event(
                            client,
                            targets,
                            event,
                            state,
                            page_alive=alive,
                            defer_draft=True,
                        )
                        schedule_draft_flush()
                    else:
                        if is_terminal_run_event(event) or event.event_type == "final_message":
                            flush_draft_now()
                        _render_run_event(
                            client,
                            targets,
                            event,
                            state,
                            page_alive=alive,
                        )
                    if event.sequence > cursor:
                        cursor = event.sequence
                        last_rendered_sequence[run_id] = cursor
                        _append_local_event(session, event)
                        if _should_refresh_sidebar(event):
                            refresh_sidebar_safe()
                    if is_terminal_run_event(event):
                        pending = pending_messages_for(session.sid)
                        if pending:
                            pending_messages_by_session[session.sid] = [
                                _mark_pending_ready(item) for item in pending
                            ]
                            if session.sid == active_sid:
                                render_pending_status()
                        schedule_successor_watch(session.sid, run_id)
                        return
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.exception("chat run subscription failed for %s", run_id)
                safe_notify(f"运行事件订阅失败：{exc}", type="negative")
            finally:
                if state.flush_task is not None and not state.flush_task.done():
                    state.flush_task.cancel()
                state.flush_task = None
                await subscription.aclose()

        def start_subscription(session: _ChatSession, snapshot: RunSnapshot) -> None:
            nonlocal subscription_task, subscribed_run_id
            if not alive():
                return
            cancel_current_subscription()
            subscribed_run_id = snapshot.run_id
            run_started_at.setdefault(snapshot.run_id, time.monotonic())
            subscription_task = track_page_task(
                asyncio.create_task(
                    consume_run_events(session, snapshot),
                    name=f"chat-page-subscription-{snapshot.run_id}",
                )
            )

        def activate_session(sid: str) -> None:
            nonlocal active_sid
            if not alive() or sid not in sessions:
                return
            if active_sid == sid:
                current = active_run(sid)
                if current is not None and subscribed_run_id != current.run_id:
                    start_subscription(sessions[sid], current)
                return
            if active_sid and active_sid in sessions:
                old_container = sessions[active_sid].container
                if not getattr(old_container, "is_deleted", False):
                    old_container.visible = False
            sessions[sid].container.visible = True
            active_sid = sid
            _set_card_visible(client, plan_card, False, page_alive=alive)
            _set_card_visible(client, progress_card, False, page_alive=alive)
            current = active_run(sid)
            if current is None:
                cancel_current_subscription()
                if manager.has_pending_successor(sid):
                    schedule_successor_watch(sid, "")
            else:
                start_subscription(sessions[sid], current)
            render_pending_status()
            refresh_sidebar_safe()

        def create_session(name: str = "", focus: bool = True) -> _ChatSession:
            session = _ChatSession(name=name, repository=repository)
            sessions[session.sid] = session
            session.transcript.move(session.container)
            session.container.move(chat_stack)
            session.container.visible = False
            session.save()
            refresh_sidebar_safe()
            if focus:
                activate_session(session.sid)
            return session

        async def close_session(sid: str) -> None:
            nonlocal active_sid
            if not alive():
                return
            if len(sessions) <= 1:
                safe_notify("Cannot close the last session.", type="warning")
                return
            if run_is_busy(sid) or manager.has_pending_successor(sid):
                safe_notify(
                    "当前会话仍在运行或停止中，请等待任务结束后再删除。",
                    type="warning",
                )
                return
            session = sessions.get(sid)
            if session is None:
                return
            try:
                deleted = await manager.delete_session(session.sid)
            except SessionDeletionBlockedError:
                safe_notify(
                    "当前会话仍在运行或停止中，请等待任务结束后再删除。",
                    type="warning",
                )
                return
            except Exception as exc:
                logger.exception("chat session delete failed for %s", sid)
                safe_notify(f"删除会话失败：{exc}", type="negative")
                return
            if not deleted:
                safe_notify("会话不存在或已删除。", type="warning")
                return
            sessions.pop(sid)
            pending_messages_by_session.pop(sid, None)
            if not alive():
                return
            if not getattr(session.container, "is_deleted", False):
                session.container.clear()
                session.container.delete()
            if active_sid == sid:
                activate_session(next(iter(sessions)))
            else:
                refresh_sidebar_safe()

        def render_pending_status() -> None:
            if not alive() or pending_card is None or pending_label is None:
                return
            pending_messages = current_pending_messages()
            if not pending_messages:
                _set_card_visible(client, queue_badge, False, page_alive=alive)
                _set_card_visible(client, pending_card, False, page_alive=alive)
                _set_card_visible(client, pending_send_button, False, page_alive=alive)
                return
            status = _pending_message_status(
                pending_messages[0], queue_depth=len(pending_messages)
            )
            if _ui_target_alive(client, queue_badge, alive):
                queue_badge.set_text(status.badge)
                queue_badge.visible = True
                queue_badge.update()
            if _ui_target_alive(client, pending_label, alive):
                pending_label.set_text(status.detail)
            _set_card_visible(client, pending_card, True, page_alive=alive)
            _set_card_visible(
                client,
                pending_send_button,
                status.can_send and not run_is_busy(active_sid),
                page_alive=alive,
            )

        def render_run_timer() -> None:
            if not alive() or run_timer_badge is None:
                return
            current = active_run()
            if current is None:
                _set_card_visible(client, run_timer_badge, False, page_alive=alive)
                return
            started = run_started_at.setdefault(current.run_id, time.monotonic())
            elapsed = int(time.monotonic() - started)
            label = "停止中" if current.status is RunStatus.CANCELLING else "运行"
            if _ui_target_alive(client, run_timer_badge, alive):
                run_timer_badge.set_text(f"{label} {_format_elapsed(elapsed)}")
                run_timer_badge.visible = True
                run_timer_badge.update()

        def cancel_pending_message() -> None:
            if not alive() or not active_sid:
                return
            pending_messages = current_pending_messages()
            if not pending_messages:
                return
            pending_messages.pop(0)
            render_pending_status()
            safe_notify("已撤销当前待发送消息。")

        async def send_pending_message() -> None:
            if not alive() or not active_sid:
                return
            pending_messages = current_pending_messages()
            if not pending_messages:
                return
            pending = pending_messages[0]
            sid = _pending_message_session(pending, active_sid)
            if run_is_busy(sid):
                safe_notify("当前任务仍在运行，稍后再确认发送。", type="warning")
                return
            pending_messages.pop(0)
            if sid == active_sid:
                message.value = pending.content
                render_pending_status()
            await run_send_handler(
                from_pending=True,
                session_id=sid,
                content=pending.content,
                level=pending.level,
            )

        async def stop_active_run() -> None:
            if not alive():
                return
            current = active_run()
            if current is None:
                safe_notify("当前没有正在运行的任务。", type="info")
                return
            snapshot = await manager.request_cancel(current.run_id)
            if snapshot is not None:
                render_run_timer()
                safe_notify("正在停止，当前工具调用结束后生效。", type="warning")

        async def run_send_handler(
            *,
            from_pending: bool,
            session_id: str | None = None,
            content: str | None = None,
            level: str | None = None,
        ) -> None:
            if not alive():
                return
            sid = session_id or active_sid
            if not sid or sid not in sessions:
                safe_notify("No active session.", type="warning")
                return
            pending_messages = pending_messages_for(sid)
            if pending_messages and pending_messages[0].ready_to_send and not from_pending:
                safe_notify("请先发送或撤销待处理消息。", type="warning")
                return
            requested_content = (content if content is not None else message.value or "").strip()
            if not requested_content:
                safe_notify("Enter a message first.", type="warning")
                return
            resolved_level = level or INTERRUPT_LABELS.get(interrupt_select.value, "guide")
            current = active_run(sid)
            if current is not None:
                if resolved_level == "interrupt":
                    if sid == active_sid:
                        message.value = ""
                    try:
                        old = await manager.interrupt_and_start(sid, requested_content)
                    except SuccessorAlreadyPendingError:
                        safe_notify("已有待执行的打断消息，请等待它完成。", type="warning")
                        return
                    except RunAlreadyActiveError as exc:
                        safe_notify(str(exc), type="warning")
                        return
                    if sid == active_sid:
                        render_pending_status()
                    safe_notify("旧任务正在停止，新消息将在停止后启动。", type="warning")
                    if old.successor_run_id is not None:
                        schedule_successor_watch(sid, old.run_id)
                    return
                pending = _build_pending_message(resolved_level, requested_content)
                if pending is None:
                    safe_notify("Enter a message first.", type="warning")
                    return
                pending_messages.append(replace(pending, session_id=sid))
                if sid == active_sid:
                    message.value = ""
                    render_pending_status()
                safe_notify(
                    "已排队，当前任务完成后等待你确认发送。"
                    if resolved_level == "queue"
                    else "已加入引导，当前任务完成后等待你确认发送。"
                )
                return

            if sid == active_sid:
                message.value = ""
            try:
                snapshot = await manager.start_run(sid, requested_content)
            except RunAlreadyActiveError as exc:
                safe_notify(str(exc), type="warning")
                return
            run_started_at[snapshot.run_id] = time.monotonic()
            if sid == active_sid:
                start_subscription(sessions[sid], snapshot)
                render_pending_status()

        async def send_handler() -> None:
            await run_send_handler(from_pending=False)

        sessions: dict[str, _ChatSession] = {}

        with ui.row().classes("w-full gap-0 flex-1").style("min-height: 0"):
            with ui.column().classes("border-r bg-gray-50/50").style(
                "width: 220px; flex-shrink: 0"
            ):
                with ui.row().classes("w-full items-center justify-between p-3 border-b"):
                    ui.label("Sessions").classes(
                        "text-sm font-semibold text-gray-500 uppercase tracking-wide"
                    )
                    ui.button(icon="add", on_click=lambda: create_session()).props(
                        "flat round size=sm dense"
                    )

                @ui.refreshable
                def sidebar_list() -> None:
                    for sid, session in sessions.items():
                        is_active = sid == active_sid
                        background = (
                            "bg-blue-50 border-l-[3px] border-l-blue-500"
                            if is_active
                            else ""
                        )
                        row = ui.row().classes(
                            f"w-full items-center gap-2 px-3 py-2.5 cursor-pointer "
                            f"hover:bg-gray-100 transition-colors {background}"
                        )
                        with row:
                            ui.icon("chat", size="sm").classes(
                                "text-blue-500" if is_active else "text-gray-400"
                            )
                            with ui.column().classes("gap-0 flex-1 min-w-0"):
                                text_class = "font-semibold text-blue-700" if is_active else ""
                                ui.label(session.name).classes(f"text-xs truncate {text_class}")
                                ui.label(session.preview or "还没有对话").classes(
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
                _refresh_fn[0] = sidebar_list.refresh
                with ui.row().classes("items-center gap-1 p-3 text-[11px] text-gray-400"):
                    settings = get_settings()
                    if settings.deepseek_api_key:
                        ui.icon("check_circle", size="xs", color="green")
                        ui.label(f"DeepSeek ({settings.deepseek_model})")
                    else:
                        ui.icon("warning", size="xs", color="orange")
                        ui.label("No LLM")

            with ui.column().classes("flex-1 flex flex-col gap-0").style("min-width: 0"):
                chat_stack = ui.column().classes("w-full flex-1 overflow-auto relative")
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
                        pending_label = ui.label("").classes("text-sm text-amber-900 grow")
                        ui.button(
                            "撤销", icon="undo", on_click=cancel_pending_message
                        ).props("flat dense color=negative")
                        pending_send_button = ui.button(
                            "发送", icon="send", on_click=send_pending_message
                        ).props("dense color=primary")
                pending_card.visible = False
                pending_send_button.visible = False

                with ui.row().classes("w-full items-end gap-2 p-4 border-t"):
                    message = (
                        ui.textarea("Type your research question...")
                        .classes("grow")
                        .props("autogrow rows=2 outlined")
                    )
                    interrupt_select = ui.select(
                        list(INTERRUPT_LABELS.keys()),
                        label="等级",
                        value="引导",
                    ).classes("w-20").props("dense outlined")
                    run_timer_badge = ui.badge("", color="green")
                    run_timer_badge.visible = False
                    queue_badge = ui.badge("", color="orange")
                    queue_badge.visible = False
                    ui.button("Stop", on_click=stop_active_run).props("color=negative flat")
                    ui.button("Send", on_click=send_handler).props("color=primary")

                ui.timer(1.0, render_run_timer)
                with ui.expansion("Advanced", icon="tune").classes("w-full px-4"):
                    with ui.row().classes("gap-4"):
                        ui.select(
                            ["auto", "stock", "etf", "stock_etf"],
                            value="auto",
                            label="Universe",
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
                            value="balanced",
                            label="Budget",
                        ).classes("w-36")
                with ui.expansion("Suggested prompts", icon="lightbulb").classes(
                    "w-full px-4"
                ):
                    with ui.row().classes("flex-wrap gap-2"):
                        for prompt in SUGGESTED_PROMPTS:
                            ui.chip(
                                prompt,
                                on_click=lambda _, text=prompt: _fill_prompt(message, text),
                            )

        for session in loaded_sessions:
            sessions[session.sid] = session
            session.transcript.move(session.container)
            session.container.move(chat_stack)
            session.container.visible = False
            session.rebuild_ui(client=client, page_alive=alive)

        if repository.last_diagnostics and alive():
            safe_notify(
                f"会话存储降级：{len(repository.last_diagnostics)} 个损坏记录待隔离",
                type="warning",
            )
        if sessions:
            refresh_sidebar_safe()
            activate_session(next(iter(sessions)))
        else:
            create_session(focus=True)

        def on_client_delete() -> None:
            nonlocal page_alive, subscription_task
            page_alive = False
            if subscription_task is not None and not subscription_task.done():
                subscription_task.cancel()
            subscription_task = None
            for task in tuple(page_tasks):
                if not task.done():
                    task.cancel()
            page_tasks.clear()

        client.on_delete(on_client_delete)

        def on_page_exception(exception: Exception) -> None:
            logger.error(
                "NiceGUI chat page exception",
                exc_info=(type(exception), exception, exception.__traceback__),
            )
            if not alive():
                return
            session = sessions.get(active_sid)
            if session is None:
                return
            try:
                _render_info_event(
                    client,
                    session.transcript,
                    f"页面错误：{exception}",
                    page_alive=alive,
                    color="red",
                )
            except Exception:
                logger.exception("failed to render chat page exception")

        client.on_exception(on_page_exception)
