"""Shared Agent runtime for Web routes."""

from __future__ import annotations

from functools import lru_cache

from qmt_agent_trader.agent.runtime import AgentRuntime, build_default_runtime
from qmt_agent_trader.core.config import get_settings


@lru_cache
def get_agent_runtime() -> AgentRuntime:
    return build_default_runtime(get_settings())
