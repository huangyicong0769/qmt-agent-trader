"""Permission model for LLM tools."""

from __future__ import annotations

from enum import StrEnum

from qmt_agent_trader.core.errors import PermissionDeniedError


class ToolCapability(StrEnum):
    READ_DATA = "READ_DATA"
    WRITE_RESEARCH = "WRITE_RESEARCH"
    RUN_BACKTEST = "RUN_BACKTEST"
    GENERATE_ORDER_PLAN = "GENERATE_ORDER_PLAN"
    SUBMIT_ORDER = "SUBMIT_ORDER"
    MODIFY_LIVE_CONFIG = "MODIFY_LIVE_CONFIG"
    DELETE_AUDIT_LOG = "DELETE_AUDIT_LOG"


LLM_ALLOWED_CAPABILITIES: frozenset[ToolCapability] = frozenset(
    {
        ToolCapability.READ_DATA,
        ToolCapability.WRITE_RESEARCH,
        ToolCapability.RUN_BACKTEST,
    }
)


APPROVED_STRATEGY_CAPABILITIES: frozenset[ToolCapability] = frozenset(
    {
        ToolCapability.READ_DATA,
        ToolCapability.RUN_BACKTEST,
        ToolCapability.GENERATE_ORDER_PLAN,
    }
)


def assert_llm_tool_allowed(capability: ToolCapability) -> None:
    if capability not in LLM_ALLOWED_CAPABILITIES:
        raise PermissionDeniedError(f"LLM is not allowed to use {capability}")
