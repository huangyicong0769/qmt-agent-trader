import pytest

from qmt_agent_trader.agent.permissions import ToolCapability, assert_llm_tool_allowed
from qmt_agent_trader.core.errors import PermissionDeniedError


def test_llm_cannot_submit_order() -> None:
    with pytest.raises(PermissionDeniedError):
        assert_llm_tool_allowed(ToolCapability.SUBMIT_ORDER)


def test_llm_can_run_backtest() -> None:
    assert_llm_tool_allowed(ToolCapability.RUN_BACKTEST)
