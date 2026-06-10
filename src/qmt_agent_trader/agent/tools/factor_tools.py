"""Agent factor tools."""

from __future__ import annotations

from qmt_agent_trader.agent.permissions import ToolCapability

CAPABILITY = ToolCapability.WRITE_RESEARCH


def propose_factor(name: str) -> dict[str, str]:
    return {"name": name, "status": "draft"}
