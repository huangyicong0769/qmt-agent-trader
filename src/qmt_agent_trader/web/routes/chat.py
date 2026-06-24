"""Chat session API routes."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from qmt_agent_trader.core.ids import shanghai_now_iso
from qmt_agent_trader.web.event_bus import AgentEvent, AgentEventType, event_bus
from qmt_agent_trader.web.schemas import (
    ChatMessage,
    ChatSession,
    CreateChatSessionRequest,
    SendMessageRequest,
)

router = APIRouter()

_sessions: dict[str, ChatSession] = {}


@router.post("/sessions", response_model=ChatSession)
async def create_session(request: CreateChatSessionRequest) -> ChatSession:
    session = ChatSession(
        title=request.title or "New research chat",
        mode=request.mode,
        context=request.context,
    )
    _sessions[session.session_id] = session
    await event_bus.publish(
        AgentEvent(
            run_id=session.session_id,
            event_type=AgentEventType.RUN_STARTED,
            title="Chat session created",
            message=session.title,
            payload={"mode": session.mode},
        )
    )
    return session


@router.get("/sessions", response_model=list[ChatSession])
async def list_sessions() -> list[ChatSession]:
    return sorted(_sessions.values(), key=lambda session: session.updated_at, reverse=True)


@router.get("/sessions/{session_id}", response_model=ChatSession)
async def get_session(session_id: str) -> ChatSession:
    return _get_session_or_404(session_id)


@router.post("/sessions/{session_id}/messages", response_model=ChatSession)
async def send_message(session_id: str, request: SendMessageRequest) -> ChatSession:
    session = _get_session_or_404(session_id)
    user_message = ChatMessage(
        session_id=session_id,
        role="user",
        content=request.content,
        metadata=request.metadata,
    )
    assistant_message = ChatMessage(
        session_id=session_id,
        role="assistant",
        content=(
            "Agent chat execution is wired for the Studio UI. "
            "Full LLM orchestration will be enabled in a later phase."
        ),
        metadata={"stub": True},
    )
    session.messages.extend([user_message, assistant_message])
    session.updated_at = shanghai_now_iso()
    await event_bus.publish(
        AgentEvent(
            run_id=session_id,
            event_type=AgentEventType.LLM_MESSAGE,
            title="Chat response",
            message=assistant_message.content,
            payload={"message_id": assistant_message.message_id, "stub": True},
        )
    )
    return session


def _get_session_or_404(session_id: str) -> ChatSession:
    session = _sessions.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="chat session not found")
    return session
