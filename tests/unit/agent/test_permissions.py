"""Tests for agent.permissions — both legacy and new permission systems."""

from __future__ import annotations

import pytest

from qmt_agent_trader.agent.permissions import (
    PermissionLevel,
    ToolCapability,
    assert_llm_tool_allowed,
    can_llm_call,
    require_permission,
)
from qmt_agent_trader.core.errors import PermissionDeniedError

# ── Legacy ToolCapability tests ──────────────────────────────────────────────


def test_legacy_llm_can_read_data() -> None:
    assert_llm_tool_allowed(ToolCapability.READ_DATA)


def test_legacy_llm_can_write_research() -> None:
    assert_llm_tool_allowed(ToolCapability.WRITE_RESEARCH)


def test_legacy_llm_cannot_submit_order() -> None:
    with pytest.raises(PermissionDeniedError):
        assert_llm_tool_allowed(ToolCapability.SUBMIT_ORDER)


def test_legacy_llm_cannot_modify_live_config() -> None:
    with pytest.raises(PermissionDeniedError):
        assert_llm_tool_allowed(ToolCapability.MODIFY_LIVE_CONFIG)


def test_legacy_llm_cannot_delete_audit() -> None:
    with pytest.raises(PermissionDeniedError):
        assert_llm_tool_allowed(ToolCapability.DELETE_AUDIT_LOG)


# ── New PermissionLevel tests ────────────────────────────────────────────────


def test_can_llm_call_read_only() -> None:
    assert can_llm_call(PermissionLevel.READ_ONLY) is True


def test_can_llm_call_research_write() -> None:
    assert can_llm_call(PermissionLevel.RESEARCH_WRITE) is True


def test_can_llm_call_code_generation() -> None:
    assert can_llm_call(PermissionLevel.CODE_GENERATION) is True


def test_can_llm_call_backtest_execute() -> None:
    assert can_llm_call(PermissionLevel.BACKTEST_EXECUTE) is True


def test_can_llm_call_approval_required() -> None:
    assert can_llm_call(PermissionLevel.APPROVAL_REQUIRED) is False


def test_can_llm_call_forbidden_to_llm() -> None:
    assert can_llm_call(PermissionLevel.FORBIDDEN_TO_LLM) is False


# ── require_permission tests ─────────────────────────────────────────────────


def test_require_permission_read_only_llm() -> None:
    require_permission(PermissionLevel.READ_ONLY, requested_by_llm=True)


def test_require_permission_approval_required_llm_raises() -> None:
    with pytest.raises(PermissionDeniedError):
        require_permission(PermissionLevel.APPROVAL_REQUIRED, requested_by_llm=True)


def test_require_permission_forbidden_to_llm_raises() -> None:
    with pytest.raises(PermissionDeniedError):
        require_permission(PermissionLevel.FORBIDDEN_TO_LLM, requested_by_llm=True)


def test_require_permission_human_can_approval_required() -> None:
    require_permission(PermissionLevel.APPROVAL_REQUIRED, requested_by_llm=False)


def test_require_permission_human_cannot_forbidden() -> None:
    with pytest.raises(PermissionDeniedError):
        require_permission(PermissionLevel.FORBIDDEN_TO_LLM, requested_by_llm=False)


# ── Direct enum value mappings ───────────────────────────────────────────────


@pytest.mark.parametrize(
    "level, expected",
    [
        (PermissionLevel.READ_ONLY, True),
        (PermissionLevel.RESEARCH_WRITE, True),
        (PermissionLevel.CODE_GENERATION, True),
        (PermissionLevel.BACKTEST_EXECUTE, True),
        (PermissionLevel.APPROVAL_REQUIRED, False),
        (PermissionLevel.FORBIDDEN_TO_LLM, False),
    ],
)
def test_permission_level_llm_callable(level: PermissionLevel, expected: bool) -> None:
    assert can_llm_call(level) is expected
