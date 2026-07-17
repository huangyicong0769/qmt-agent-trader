"""Shared Agent runtime for Web routes."""

from __future__ import annotations

from functools import lru_cache

from qmt_agent_trader.agent.orchestrator import AgentOrchestrator
from qmt_agent_trader.agent.runtime import AgentRuntime, build_default_runtime
from qmt_agent_trader.core.config import get_settings
from qmt_agent_trader.web.chat_repository import build_chat_session_repository
from qmt_agent_trader.web.chat_run_manager import ChatRunManager


@lru_cache
def get_agent_runtime() -> AgentRuntime:
    return build_default_runtime(get_settings())


@lru_cache
def get_chat_run_manager() -> ChatRunManager:
    """Return the one application-scoped chat run manager."""
    return ChatRunManager(
        orchestrator=AgentOrchestrator(runtime=get_agent_runtime()),
        repository=build_chat_session_repository(get_settings()),
    )
