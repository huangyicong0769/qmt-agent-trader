"""Safe read-only research context tools for LLM workflows."""

from __future__ import annotations

from typing import Any

from qmt_agent_trader.agent.llm_client import DeepSeekTool
from qmt_agent_trader.agent.permissions import (
    LLM_ALLOWED_CAPABILITIES,
    ToolCapability,
)
from qmt_agent_trader.agent.tool_registry import ToolDefinition, ToolRegistry

IMPLEMENTED_DAILY_FACTORS = [
    "momentum_20d",
    "momentum_60d",
    "reversal_5d",
    "volatility_20d",
    "turnover_20d",
    "amount_zscore_20d",
]


def get_research_context(universe: str) -> dict[str, Any]:
    return {
        "universe": [item.strip() for item in universe.split(",") if item.strip()],
        "frequency": "daily",
        "historical_data_source": "Tushare Pro via local data lake",
        "latest_market_source": "Windows QMT Gateway when configured",
        "implemented_daily_factors": IMPLEMENTED_DAILY_FACTORS,
        "trade_state_inputs": [
            "suspend_d",
            "stk_limit",
            "stock_basic.name",
            "namechange",
        ],
        "required_backtest_constraints": [
            "T+1 execution",
            "suspended symbols cannot trade",
            "limit-up symbols cannot be bought",
            "limit-down symbols cannot be sold",
            "ST symbols are filtered by default",
            "signal date and execution date must be separated",
        ],
        "llm_allowed_capabilities": sorted(
            capability.value for capability in LLM_ALLOWED_CAPABILITIES
        ),
        "llm_forbidden_capabilities": [
            ToolCapability.SUBMIT_ORDER.value,
            ToolCapability.MODIFY_LIVE_CONFIG.value,
            ToolCapability.DELETE_AUDIT_LOG.value,
        ],
    }


def build_research_context_tool() -> DeepSeekTool:
    registry = ToolRegistry()
    registry.register(
        ToolDefinition(
            name="get_research_context",
            capability=ToolCapability.READ_DATA,
            fn=get_research_context,
        )
    )
    return DeepSeekTool(
        name="get_research_context",
        description=(
            "Return the local daily research data sources, implemented factors, "
            "backtest constraints, and LLM permission boundaries."
        ),
        parameters={
            "type": "object",
            "properties": {
                "universe": {
                    "type": "string",
                    "description": "Comma-separated universe names, for example stock,etf.",
                }
            },
            "required": ["universe"],
            "additionalProperties": False,
        },
        fn=lambda universe: registry.call_as_llm("get_research_context", universe=universe),
    )
