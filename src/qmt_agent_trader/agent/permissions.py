"""Permission model for LLM tools and Agent operations.

This module defines two permission systems that coexist:
1.  `ToolCapability` (original): used by the existing tool registry and agent runtime.
2.  `PermissionLevel` (new): richer six-level system for the expanded Agent subsystem.

The two are bridged so that the original `assert_llm_tool_allowed` still works,
while new tools can use the finer-grained `PermissionLevel`.
"""

from __future__ import annotations

from enum import StrEnum

from qmt_agent_trader.core.errors import PermissionDeniedError

# ── Original capability system (kept for backward compatibility) ─────────────

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


# ── New six-level permission system ──────────────────────────────────────────

class PermissionLevel(StrEnum):
    READ_ONLY = "READ_ONLY"
    RESEARCH_WRITE = "RESEARCH_WRITE"
    CODE_GENERATION = "CODE_GENERATION"
    BACKTEST_EXECUTE = "BACKTEST_EXECUTE"
    APPROVAL_REQUIRED = "APPROVAL_REQUIRED"
    FORBIDDEN_TO_LLM = "FORBIDDEN_TO_LLM"


class ToolCallMode(StrEnum):
    AUTONOMOUS_AGENT = "AUTONOMOUS_AGENT"
    TRUSTED_INTERNAL_WORKFLOW = "TRUSTED_INTERNAL_WORKFLOW"


_AUTONOMOUS_CALLABLE: frozenset[PermissionLevel] = frozenset(
    {
        PermissionLevel.READ_ONLY,
        PermissionLevel.RESEARCH_WRITE,
        PermissionLevel.CODE_GENERATION,
        PermissionLevel.BACKTEST_EXECUTE,
    }
)


_TRUSTED_INTERNAL_CALLABLE: frozenset[PermissionLevel] = frozenset(
    {*_AUTONOMOUS_CALLABLE, PermissionLevel.APPROVAL_REQUIRED}
)


def can_call_tool(permission: PermissionLevel, mode: ToolCallMode) -> bool:
    """Return True when a runtime call mode may execute this permission level."""
    if permission == PermissionLevel.FORBIDDEN_TO_LLM:
        return False
    if mode == ToolCallMode.TRUSTED_INTERNAL_WORKFLOW:
        return permission in _TRUSTED_INTERNAL_CALLABLE
    return permission in _AUTONOMOUS_CALLABLE


def can_llm_call(permission: PermissionLevel) -> bool:
    """Backward-compatible alias for autonomous-agent callable tools."""
    return can_call_tool(permission, ToolCallMode.AUTONOMOUS_AGENT)


def require_permission(
    tool_permission: PermissionLevel,
    *,
    requested_by_llm: bool = True,
    call_mode: ToolCallMode | None = None,
    tool_name: str | None = None,
) -> None:
    """Raise `PermissionDeniedError` if the caller is not authorised.

    Web, chat, and LLM tool calls are all autonomous-agent calls. Only explicit
    internal workflows may run approval-required review tools.
    """
    label = f" ({tool_name})" if tool_name else ""
    mode = call_mode or (
        ToolCallMode.AUTONOMOUS_AGENT
        if requested_by_llm
        else ToolCallMode.TRUSTED_INTERNAL_WORKFLOW
    )
    if can_call_tool(tool_permission, mode):
        return
    if tool_permission == PermissionLevel.FORBIDDEN_TO_LLM:
        raise PermissionDeniedError(
            f"tool{label} is FORBIDDEN_TO_LLM and cannot be invoked by the agent runtime"
        )
    if tool_permission == PermissionLevel.APPROVAL_REQUIRED:
        raise PermissionDeniedError(
            f"tool{label} requires trusted internal workflow context"
        )
    raise PermissionDeniedError(
        f"tool{label} at level {tool_permission.value} is not allowed for {mode.value}"
    )


# ── Bridge: map new PermissionLevel → old ToolCapability (best-effort) ───────

_PERMISSION_TO_CAPABILITY: dict[PermissionLevel, ToolCapability] = {
    PermissionLevel.READ_ONLY: ToolCapability.READ_DATA,
    PermissionLevel.RESEARCH_WRITE: ToolCapability.WRITE_RESEARCH,
    PermissionLevel.CODE_GENERATION: ToolCapability.WRITE_RESEARCH,
    PermissionLevel.BACKTEST_EXECUTE: ToolCapability.RUN_BACKTEST,
    PermissionLevel.APPROVAL_REQUIRED: ToolCapability.GENERATE_ORDER_PLAN,
    PermissionLevel.FORBIDDEN_TO_LLM: ToolCapability.SUBMIT_ORDER,
}


def to_capability(level: PermissionLevel) -> ToolCapability:
    """Best-effort mapping from the new level to the legacy capability."""
    return _PERMISSION_TO_CAPABILITY.get(level, ToolCapability.READ_DATA)
