"""Chat session and application-owned run API routes."""

from __future__ import annotations

import json
from collections.abc import AsyncGenerator
from functools import lru_cache

from fastapi import APIRouter, HTTPException, Request, Response
from fastapi.responses import StreamingResponse

from qmt_agent_trader.core.ids import shanghai_now_iso
from qmt_agent_trader.web.chat_repository import (
    ChatSessionRepository,
    build_chat_session_repository,
)
from qmt_agent_trader.web.chat_run_manager import (
    ChatRunManager,
    RunAlreadyActiveError,
    RunSnapshot,
    SuccessorAlreadyPendingError,
)
from qmt_agent_trader.web.runtime import get_agent_runtime, get_chat_run_manager
from qmt_agent_trader.web.schemas import (
    ChatMessage,
    ChatSession,
    CreateChatSessionRequest,
    SendMessageRequest,
    StartChatRunRequest,
)

router = APIRouter()


def _get_run_manager() -> ChatRunManager:
    return get_chat_run_manager()


@router.post("/sessions", response_model=ChatSession)
async def create_session(request: CreateChatSessionRequest) -> ChatSession:
    session = ChatSession(
        title=request.title or "New research chat",
        context=request.context,
    )
    return get_chat_session_repository().create(session)


@router.get("/sessions", response_model=list[ChatSession])
async def list_sessions(response: Response) -> list[ChatSession]:
    repository = get_chat_session_repository()
    sessions = repository.list()
    response.headers["X-Storage-Status"] = (
        "DEGRADED" if repository.last_diagnostics else "OK"
    )
    if repository.last_diagnostics:
        response.headers["X-Storage-Diagnostics-Count"] = str(
            len(repository.last_diagnostics)
        )
    return sessions


@router.get("/sessions/{session_id}", response_model=ChatSession)
async def get_session(session_id: str) -> ChatSession:
    return _get_session_or_404(session_id)


@router.post("/sessions/{session_id}/messages")
async def send_message(session_id: str, request: SendMessageRequest) -> dict[str, object]:
    """Store a user message without starting an agent run."""
    session = _get_session_or_404(session_id)
    user_message = ChatMessage(
        session_id=session_id,
        role="user",
        content=request.content,
        metadata=request.metadata or {},
    )
    llm_configured = get_agent_runtime().settings.deepseek_api_key is not None
    session = get_chat_session_repository().update(
        session_id,
        lambda current: current.model_copy(
            update={
                "messages": [*current.messages, user_message],
                "updated_at": shanghai_now_iso(),
            }
        ),
    )
    return {
        "session_id": session.session_id,
        "run_id": session.session_id,
        "message_id": user_message.message_id,
        "message": user_message,
        "llm_configured": llm_configured,
    }


@router.post("/sessions/{session_id}/runs")
async def create_run(session_id: str, request: StartChatRunRequest) -> dict[str, object]:
    session = _get_session_or_404(session_id)
    message = (request.message or request.content or "").strip()
    if not message:
        raise HTTPException(status_code=400, detail="No message to execute")
    manager = _get_run_manager()
    try:
        if request.interrupt:
            snapshot = await manager.interrupt_and_start(
                session_id,
                message,
                persist_user_message=not _has_unclaimed_user_message(session, message),
            )
        else:
            snapshot = await manager.start_run(
                session_id,
                message,
                persist_user_message=not _has_unclaimed_user_message(session, message),
            )
    except (RunAlreadyActiveError, SuccessorAlreadyPendingError) as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return snapshot.to_dict()


@router.get("/runs/{run_id}")
async def get_run(run_id: str) -> dict[str, object]:
    snapshot = _get_run_manager().get_run(run_id)
    if snapshot is None:
        raise HTTPException(status_code=404, detail="chat run not found")
    return snapshot.to_dict()


@router.get("/runs/{run_id}/events")
async def stream_run_events(run_id: str, after_sequence: int = 0) -> StreamingResponse:
    manager = _get_run_manager()
    if manager.get_run(run_id) is None:
        raise HTTPException(status_code=404, detail="chat run not found")

    async def event_generator() -> AsyncGenerator[str, None]:
        subscription = manager.subscribe(
            run_id,
            after_sequence=max(0, after_sequence),
        )
        try:
            async for event in subscription:
                yield f"event: {event.event_type}\n"
                yield f"data: {event.to_sse()}\n\n"
        finally:
            await subscription.aclose()

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/runs/{run_id}/cancel")
async def cancel_run(run_id: str) -> dict[str, object]:
    snapshot = await _get_run_manager().request_cancel(run_id)
    if snapshot is None:
        raise HTTPException(status_code=404, detail="chat run not found")
    return snapshot.to_dict()


@router.post("/sessions/{session_id}/execute")
async def execute_stream(session_id: str, request: Request) -> StreamingResponse:
    """Compatibility SSE endpoint backed entirely by ChatRunManager."""
    session = _get_session_or_404(session_id)
    body: dict[str, object] = {}
    try:
        raw_body = await request.json()
        if isinstance(raw_body, dict):
            body = raw_body
    except (json.JSONDecodeError, UnicodeDecodeError):
        body = {}

    raw_message = body.get("message", "") or body.get("content", "")
    message = str(raw_message).strip() if raw_message else ""
    using_session_message = False
    if not message:
        for item in reversed(session.messages):
            if item.role == "user":
                message = item.content
                using_session_message = True
                break
    elif _has_unclaimed_user_message(session, message):
        using_session_message = True
    if not message:
        raise HTTPException(status_code=400, detail="No message to execute")

    manager = _get_run_manager()
    try:
        snapshot = await manager.start_run(
            session_id,
            message,
            persist_user_message=not using_session_message,
        )
    except RunAlreadyActiveError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return _streaming_response_for_run(manager, snapshot)


def _streaming_response_for_run(
    manager: ChatRunManager,
    snapshot: RunSnapshot,
) -> StreamingResponse:
    async def event_generator() -> AsyncGenerator[str, None]:
        subscription = manager.subscribe(snapshot.run_id)
        try:
            async for event in subscription:
                yield f"event: {event.event_type}\n"
                yield f"data: {event.to_sse()}\n\n"
        finally:
            await subscription.aclose()

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


def _get_session_or_404(session_id: str) -> ChatSession:
    session = get_chat_session_repository().get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="chat session not found")
    return session


def _has_unclaimed_user_message(session: ChatSession, message: str) -> bool:
    """Recognize the legacy /messages -> /execute handoff without double-write."""
    for item in reversed(session.messages):
        if item.role != "user":
            continue
        return item.content == message and "run_id" not in item.metadata
    return False


@lru_cache
def get_chat_session_repository() -> ChatSessionRepository:
    return build_chat_session_repository()
