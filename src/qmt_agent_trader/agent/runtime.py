"""Default safe tool runtime for the research agent."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from qmt_agent_trader.agent.llm_client import DeepSeekClient, DeepSeekToolLoopResult
from qmt_agent_trader.agent.permissions import ToolCapability
from qmt_agent_trader.agent.tool_registry import ToolDefinition, ToolRegistry
from qmt_agent_trader.agent.tools.backtest_tools import (
    plan_sensitivity_analysis,
    run_factor_rank_sensitivity,
)
from qmt_agent_trader.agent.tools.research_context import get_research_context
from qmt_agent_trader.backtest.service import (
    compare_backtest_reports,
    run_backtest_report,
)
from qmt_agent_trader.broker.remote_client import RemoteQMTBrokerClient
from qmt_agent_trader.core.config import Settings, get_settings
from qmt_agent_trader.data.bars import load_daily_bars
from qmt_agent_trader.data.storage import DataLake
from qmt_agent_trader.factors.service import (
    compute_factor_to_lake,
    validate_factor,
)
from qmt_agent_trader.strategy.approval import read_approval_file


@dataclass
class AgentRuntime:
    settings: Settings
    lake: DataLake
    reports_dir: Path
    approvals_dir: Path
    broker_client: RemoteQMTBrokerClient | None = None

    def registry(self) -> ToolRegistry:
        return build_default_tool_registry(self)

    def call_tool(self, tool_name: str, **kwargs: Any) -> Any:
        return self.registry().call_as_llm(tool_name, **kwargs)

    def ask(self, prompt: str, *, max_rounds: int = 4) -> DeepSeekToolLoopResult:
        if self.settings.deepseek_api_key is None:
            raise ValueError("DEEPSEEK_API_KEY is required for agent ask")
        client = DeepSeekClient(
            api_key=self.settings.deepseek_api_key.get_secret_value(),
            base_url=self.settings.deepseek_base_url,
            model=self.settings.deepseek_model,
        )
        return client.run_tool_loop(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are the QMT research agent. Use tools for local facts. "
                        "You may read data, write research artifacts, and run simulated "
                        "backtests. You must not submit live orders, modify live config, "
                        "or bypass approvals."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            tools=self.registry().deepseek_tools_for_llm(),
            max_rounds=max_rounds,
        )


def build_default_runtime(
    settings: Settings | None = None,
    *,
    broker_client: RemoteQMTBrokerClient | None = None,
) -> AgentRuntime:
    resolved = settings or get_settings()
    lake = DataLake(
        root=resolved.resolved_data_dir / "lake",
        duckdb_path=resolved.resolved_data_dir / "qmt_agent_trader.duckdb",
    )
    return AgentRuntime(
        settings=resolved,
        lake=lake,
        reports_dir=resolved.project_root / "reports" / "backtests",
        approvals_dir=resolved.project_root / "approvals",
        broker_client=broker_client or _optional_broker_client(resolved),
    )


def build_default_tool_registry(runtime: AgentRuntime) -> ToolRegistry:
    registry = ToolRegistry()

    registry.register(
        ToolDefinition(
            name="get_research_context",
            capability=ToolCapability.READ_DATA,
            description="Return local research capabilities, constraints, and LLM boundaries.",
            parameters=_object_schema(
                {"universe": {"type": "string", "description": "Comma-separated universe names."}},
                required=["universe"],
            ),
            fn=get_research_context,
        )
    )
    registry.register(
        ToolDefinition(
            name="list_datasets",
            capability=ToolCapability.READ_DATA,
            description="List datasets in the local DuckDB/Parquet data lake.",
            parameters=_object_schema(
                {
                    "layer": {
                        "type": "string",
                        "description": "Optional layer: raw, bronze, silver, or gold.",
                    },
                    "prefix": {"type": "string", "description": "Optional dataset prefix."},
                }
            ),
            fn=lambda layer=None, prefix=None: list_datasets(
                runtime.lake, layer=layer, prefix=prefix
            ),
        )
    )
    registry.register(
        ToolDefinition(
            name="summarize_daily_bars",
            capability=ToolCapability.READ_DATA,
            description="Summarize canonical daily bars and trade-state counts.",
            parameters=_object_schema(
                {
                    "start": {"type": "string", "description": "Optional start date."},
                    "end": {"type": "string", "description": "Optional end date."},
                }
            ),
            fn=lambda start=None, end=None: summarize_daily_bars(
                runtime.lake, start=start, end=end
            ),
        )
    )
    registry.register(
        ToolDefinition(
            name="list_factors",
            capability=ToolCapability.READ_DATA,
            description="List built-in daily factor names currently implemented.",
            parameters=_object_schema({}),
            fn=list_factors,
        )
    )
    registry.register(
        ToolDefinition(
            name="compute_factor",
            capability=ToolCapability.WRITE_RESEARCH,
            description="Compute a built-in factor for one date and write it to the gold layer.",
            parameters=_object_schema(
                {
                    "name": {"type": "string", "description": "Factor name."},
                    "date": {"type": "string", "description": "Target date."},
                },
                required=["name", "date"],
            ),
            fn=lambda name, date: compute_factor_to_lake(
                runtime.lake, name=name, date=date
            ).as_dict(),
        )
    )
    registry.register(
        ToolDefinition(
            name="validate_factor",
            capability=ToolCapability.RUN_BACKTEST,
            description="Validate a built-in factor over a date range.",
            parameters=_object_schema(
                {
                    "name": {"type": "string", "description": "Factor name."},
                    "start": {"type": "string", "description": "Start date."},
                    "end": {"type": "string", "description": "End date."},
                },
                required=["name", "start", "end"],
            ),
            fn=lambda name, start, end: validate_factor(
                runtime.lake, name=name, start=start, end=end
            ).as_dict(),
        )
    )
    registry.register(
        ToolDefinition(
            name="run_backtest",
            capability=ToolCapability.RUN_BACKTEST,
            description="Run a daily T+1 single-symbol simulated backtest and persist a report.",
            parameters=_object_schema(
                {
                    "symbol": {"type": "string", "description": "Optional symbol."},
                    "signal_date": {"type": "string", "description": "Optional signal date."},
                    "quantity": {"type": "integer", "description": "Order quantity."},
                }
            ),
            fn=lambda symbol=None, signal_date=None, quantity=100: run_backtest_report(
                runtime.lake,
                reports_dir=runtime.reports_dir,
                symbol=symbol,
                signal_date=signal_date,
                quantity=quantity,
            ).as_dict(),
        )
    )
    registry.register(
        ToolDefinition(
            name="compare_backtests",
            capability=ToolCapability.RUN_BACKTEST,
            description="Compare recent persisted backtest reports.",
            parameters=_object_schema(
                {"limit": {"type": "integer", "description": "Maximum number of runs."}}
            ),
            fn=lambda limit=10: compare_backtest_reports(runtime.reports_dir, limit=limit),
        )
    )
    registry.register(
        ToolDefinition(
            name="plan_sensitivity_analysis",
            capability=ToolCapability.RUN_BACKTEST,
            description=(
                "Build a robustness scenario matrix for cost, slippage, delay, top_n, "
                "and position-cap sensitivity. This plans scenarios but does not approve "
                "or execute live trading."
            ),
            parameters=_object_schema(
                {
                    "cost_multipliers": {
                        "type": "array",
                        "items": {"type": "number"},
                        "description": "Optional cost multipliers, e.g. [1, 2, 3].",
                    },
                    "slippage_bps": {
                        "type": "array",
                        "items": {"type": "number"},
                        "description": "Optional slippage assumptions in basis points.",
                    },
                    "execution_delay_days": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": "Optional execution delay values in trading days.",
                    },
                    "top_n": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": "Optional Top-N strategy parameter values.",
                    },
                    "max_single_position_pct": {
                        "type": "array",
                        "items": {"type": "number"},
                        "description": "Optional max single position caps.",
                    },
                }
            ),
            fn=plan_sensitivity_analysis,
        )
    )
    registry.register(
        ToolDefinition(
            name="run_factor_rank_sensitivity",
            capability=ToolCapability.RUN_BACKTEST,
            description=(
                "Run a data-lake factor-rank robustness simulation across cost, slippage, "
                "execution-delay, top_n, and max-position scenarios. This is research-only "
                "and never creates approvals or order plans."
            ),
            parameters=_object_schema(
                {
                    "factor_name": {
                        "type": "string",
                        "description": "Built-in factor name such as momentum_20d.",
                    },
                    "cost_multipliers": {
                        "type": "array",
                        "items": {"type": "number"},
                    },
                    "slippage_bps": {
                        "type": "array",
                        "items": {"type": "number"},
                    },
                    "execution_delay_days": {
                        "type": "array",
                        "items": {"type": "integer"},
                    },
                    "top_n": {
                        "type": "array",
                        "items": {"type": "integer"},
                    },
                    "max_single_position_pct": {
                        "type": "array",
                        "items": {"type": "number"},
                    },
                    "initial_cash": {
                        "type": "number",
                        "description": "Initial simulation cash.",
                    },
                },
                required=["factor_name"],
            ),
            fn=lambda factor_name, cost_multipliers=None, slippage_bps=None,
            execution_delay_days=None, top_n=None, max_single_position_pct=None,
            initial_cash=1_000_000.0: run_factor_rank_sensitivity(
                runtime.lake,
                factor_name=factor_name,
                cost_multipliers=cost_multipliers,
                slippage_bps=slippage_bps,
                execution_delay_days=execution_delay_days,
                top_n=top_n,
                max_single_position_pct=max_single_position_pct,
                initial_cash=initial_cash,
            ),
        )
    )
    registry.register(
        ToolDefinition(
            name="list_strategy_approvals",
            capability=ToolCapability.READ_DATA,
            description="List local strategy approval files and paper/live flags.",
            parameters=_object_schema({}),
            fn=lambda: list_strategy_approvals(runtime.approvals_dir),
        )
    )
    registry.register(
        ToolDefinition(
            name="broker_health",
            capability=ToolCapability.READ_DATA,
            description="Check the configured Windows QMT Gateway health endpoint if available.",
            parameters=_object_schema({}),
            fn=lambda: broker_health(runtime.broker_client),
        )
    )
    return registry


def list_datasets(
    lake: DataLake, *, layer: str | None = None, prefix: str | None = None
) -> dict[str, object]:
    layers = [layer] if layer else ["raw", "bronze", "silver", "gold"]
    return {
        "layers": {
            item: lake.list_dataset_names(item, prefix=prefix)
            for item in layers
        }
    }


def summarize_daily_bars(
    lake: DataLake, *, start: str | None = None, end: str | None = None
) -> dict[str, object]:
    bars = load_daily_bars(lake, start=start, end=end)
    if bars.empty:
        return {"status": "empty", "rows": 0}
    return {
        "status": "ok",
        "rows": len(bars),
        "symbols": int(bars["symbol"].nunique()),
        "start": f"{pd.to_datetime(bars['trade_date'].min()).date():%Y%m%d}",
        "end": f"{pd.to_datetime(bars['trade_date'].max()).date():%Y%m%d}",
        "trade_state_counts": {
            "suspended": int(bars["suspended"].sum()),
            "limit_up": int(bars["limit_up"].sum()),
            "limit_down": int(bars["limit_down"].sum()),
            "st": int(bars["st"].sum()),
        },
    }


def list_factors() -> dict[str, object]:
    return {
        "factors": [
            "momentum_20d",
            "momentum_60d",
            "reversal_5d",
            "volatility_20d",
            "turnover_20d",
            "amount_zscore_20d",
        ]
    }


def list_strategy_approvals(directory: Path) -> dict[str, object]:
    if not directory.exists():
        return {"approvals": []}
    approvals = []
    for path in sorted(directory.glob("*.approval.yaml")):
        approval = read_approval_file(path)
        approvals.append(
            {
                "strategy_id": approval.strategy_id,
                "strategy_version": approval.strategy_version,
                "paper_trading_allowed": approval.paper_trading_allowed,
                "live_trading_allowed": approval.live_trading_allowed,
                "path": str(path),
            }
        )
    return {"approvals": approvals}


def broker_health(client: RemoteQMTBrokerClient | None) -> dict[str, object]:
    if client is None:
        return {"configured": False, "status": "unavailable"}
    try:
        return {"configured": True, "response": client.health()}
    except Exception as exc:
        return {"configured": True, "status": "error", "error": str(exc)}


def _optional_broker_client(settings: Settings) -> RemoteQMTBrokerClient | None:
    if settings.qmt_gateway_api_key is None or settings.qmt_gateway_hmac_secret is None:
        return None
    return RemoteQMTBrokerClient(
        base_url=settings.qmt_gateway_base_url,
        api_key=settings.qmt_gateway_api_key.get_secret_value(),
        hmac_secret=settings.qmt_gateway_hmac_secret.get_secret_value(),
    )


def _object_schema(
    properties: dict[str, dict[str, Any]], *, required: list[str] | None = None
) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": properties,
        "required": required or [],
        "additionalProperties": False,
    }
