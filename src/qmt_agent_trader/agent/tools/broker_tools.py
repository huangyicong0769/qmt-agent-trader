"""Broker tools intentionally exclude live submit for LLM usage."""

from __future__ import annotations

from qmt_agent_trader.agent.permissions import ToolCapability

CAPABILITY = ToolCapability.READ_DATA


def query_gateway_health() -> dict[str, str]:
    return {"status": "unknown"}
