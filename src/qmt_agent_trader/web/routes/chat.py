"""Chat session API routes — natural language, no forced mode selection.

Includes:
- /sessions CRUD
- /messages (message enqueue)
- /sessions/{id}/execute (SSE streaming, real LLM orchestration)
"""

from __future__ import annotations

import json
from collections.abc import AsyncGenerator

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from qmt_agent_trader.agent.orchestrator import AgentOrchestrator
from qmt_agent_trader.core.ids import new_id, shanghai_now_iso
from qmt_agent_trader.web.event_bus import AgentEvent, AgentEventType, event_bus
from qmt_agent_trader.web.runtime import get_agent_runtime
from qmt_agent_trader.web.schemas import (
    ChatMessage,
    ChatSession,
    CreateChatSessionRequest,
    SendMessageRequest,
)

router = APIRouter()

_sessions: dict[str, ChatSession] = {}

# Lazily built orchestrator (cache after first use)
_orchestrator: AgentOrchestrator | None = None


def _get_orchestrator() -> AgentOrchestrator:
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = AgentOrchestrator(runtime=get_agent_runtime())
    return _orchestrator


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
    """Store a user message.

    For real LLM execution, use the /execute SSE endpoint.
    """
    session = _get_session_or_404(session_id)
    user_message = ChatMessage(
        session_id=session_id,
        role="user",
        content=request.content,
        metadata=request.metadata or {},
    )
    orchestrator = _get_orchestrator()
    llm_configured = orchestrator.settings.deepseek_api_key is not None
    session.messages.append(user_message)
    session.updated_at = shanghai_now_iso()

    await event_bus.publish(
        AgentEvent(
            run_id=session_id,
            event_type=AgentEventType.PROGRESS,
            title="Chat message received",
            message=request.content,
            payload={
                "message_id": user_message.message_id,
                "llm_configured": llm_configured,
            },
        )
    )

    return {
        "session_id": session.session_id,
        "run_id": session.session_id,
        "message_id": user_message.message_id,
        "message": user_message,
        "llm_configured": llm_configured,
    }


@router.post("/sessions/{session_id}/execute")
async def execute_stream(session_id: str, request: Request) -> StreamingResponse:
    """SSE endpoint: real LLM orchestration with live event streaming.

    POST body: {"message": "发现低波动因子"}

    Returns: text/event-stream with JSON-encoded OrchestratorEvent per line.
    """
    session = _get_session_or_404(session_id)

    # Parse body
    body = {}
    try:
        body = await request.json()
    except Exception:
        pass

    message = body.get("message", "") or body.get("content", "")
    using_session_message = False
    if not message:
        # Use the last user message in the session
        for m in reversed(session.messages):
            if m.role == "user":
                message = m.content
                using_session_message = True
                break
    if not message:
        raise HTTPException(status_code=400, detail="No message to execute")

    run_id = new_id("run")

    if not using_session_message:
        user_msg = ChatMessage(session_id=session_id, role="user", content=message)
        session.messages.append(user_msg)
        session.updated_at = shanghai_now_iso()

    orchestrator = _get_orchestrator()

    async def event_generator() -> AsyncGenerator[str, None]:
        """Stream OrchestratorEvents as SSE."""
        try:
            async for event in orchestrator.execute_stream(
                message=message,
                run_id=run_id,
                history=[
                    {"role": m.role, "content": m.content}
                    for m in session.messages
                ],
            ):
                sse = event.to_sse()
                yield f"event: {event.type}\n"
                yield f"data: {sse}\n\n"

                # Also publish to the in-process EventBus
                await event_bus.publish(
                    AgentEvent(
                        run_id=run_id,
                        experiment_id=event.data.get("experiment_id"),
                        event_type=_to_agent_event_type(event.type),
                        title=event.type,
                        message=event.message,
                        payload=event.data,
                    )
                )

        except Exception as exc:
            error_payload = json.dumps(
                {"type": "error", "message": str(exc)}, ensure_ascii=False
            )
            yield f"event: error\ndata: {error_payload}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


def _to_agent_event_type(event_type: str) -> AgentEventType:
    """Map orchestrator event types to AgentEvent types."""
    mapping: dict[str, AgentEventType] = {
        "run_started": AgentEventType.RUN_STARTED,
        "done": AgentEventType.RUN_COMPLETED,
        "error": AgentEventType.RUN_FAILED,
        "token": AgentEventType.LLM_TOKEN_DELTA,
        "final_message": AgentEventType.LLM_TOKEN_DELTA,
        "tool_done": AgentEventType.TOOL_CALL_COMPLETED,
        "tool_start": AgentEventType.TOOL_CALL_STARTED,
        "progress": AgentEventType.PROGRESS,
    }
    return mapping.get(event_type, AgentEventType.PROGRESS)


def _get_session_or_404(session_id: str) -> ChatSession:
    session = _sessions.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="chat session not found")
    return session
