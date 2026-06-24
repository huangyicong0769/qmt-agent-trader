"""Chat session API routes — natural language, no forced mode selection."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from qmt_agent_trader.agent.router import agent_router
from qmt_agent_trader.core.ids import shanghai_now_iso
from qmt_agent_trader.web.event_bus import AgentEvent, AgentEventType, event_bus
from qmt_agent_trader.web.schemas import (
    ChatMessage,
    ChatSession,
    CreateChatSessionRequest,
    RoutingInfo,
    SendMessageRequest,
)

router = APIRouter()

_sessions: dict[str, ChatSession] = {}


@router.post("/sessions", response_model=ChatSession)
async def create_session(request: CreateChatSessionRequest) -> ChatSession:
    session = ChatSession(
        title=request.title or "New research chat",
        context=request.context,
    )
    _sessions[session.session_id] = session
    await event_bus.publish(
        AgentEvent(
            run_id=session.session_id,
            event_type=AgentEventType.RUN_STARTED,
            title="Chat session created",
            message=session.title,
        )
    )
    return session


@router.get("/sessions", response_model=list[ChatSession])
async def list_sessions() -> list[ChatSession]:
    return sorted(_sessions.values(), key=lambda s: s.updated_at, reverse=True)


@router.get("/sessions/{session_id}", response_model=ChatSession)
async def get_session(session_id: str) -> ChatSession:
    return _get_session_or_404(session_id)


@router.post("/sessions/{session_id}/messages")
async def send_message(session_id: str, request: SendMessageRequest) -> dict[str, object]:
    session = _get_session_or_404(session_id)

    # ── Agent Router: classify intent from natural language ──
    decision = agent_router.route(
        message=request.content,
        session_context=session.context,
    )

    routing_info = RoutingInfo(
        intent=decision.intent.value,
        confidence=decision.confidence,
        rationale=decision.rationale,
        required_tools=decision.required_tools,
        proposed_workflow=decision.proposed_workflow,
        parameters=decision.parameters,
        needs_user_clarification=decision.needs_user_clarification,
        clarification_question=decision.clarification_question,
    )
    session.routing_history.append(routing_info)

    user_message = ChatMessage(
        session_id=session_id,
        role="user",
        content=request.content,
        metadata=request.metadata or {},
    )

    # Build assistant response
    tools_preview = decision.required_tools[:6]
    if len(decision.required_tools) > 6:
        extra = len(decision.required_tools) - 6
        tools_preview_str = ", ".join(tools_preview) + f"... (+{extra} more)"
    else:
        tools_preview_str = ", ".join(tools_preview)

    content = (
        f"**Intent:** {decision.intent.value} (confidence: {decision.confidence:.0%})\n\n"
        f"**Plan:** {decision.rationale}\n\n"
        f"**Required tools:** {tools_preview_str or 'none'}\n\n"
        "Agent chat execution is wired for the Studio UI. "
        "Full LLM orchestration will be enabled in a later phase."
    )
    assistant_message = ChatMessage(
        session_id=session_id,
        role="assistant",
        content=content,
        metadata={"stub": True, "intent": decision.intent.value},
    )
    session.messages.extend([user_message, assistant_message])
    session.updated_at = shanghai_now_iso()

    await event_bus.publish(
        AgentEvent(
            run_id=session_id,
            event_type=AgentEventType.LLM_MESSAGE,
            title=f"Intent: {decision.intent.value}",
            message=decision.rationale,
            payload={
                "message_id": assistant_message.message_id,
                "intent": decision.intent.value,
                "confidence": decision.confidence,
                "stub": True,
            },
        )
    )

    return {
        "session_id": session.session_id,
        "run_id": session.session_id,
        "message_id": assistant_message.message_id,
        "message": assistant_message,
        "routing_decision": routing_info,
    }


def _get_session_or_404(session_id: str) -> ChatSession:
    session = _sessions.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="chat session not found")
    return session
