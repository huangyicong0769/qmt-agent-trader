"""Agent data tools."""

from __future__ import annotations

from qmt_agent_trader.agent.permissions import ToolCapability

CAPABILITY = ToolCapability.READ_DATA


def list_datasets() -> list[str]:
    return []
