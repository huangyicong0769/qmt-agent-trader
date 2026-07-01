"""Canonical strategy domain models."""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field

from qmt_agent_trader.core.ids import shanghai_now_iso
from qmt_agent_trader.core.types import ApprovalStatus


class StrategyKind(StrEnum):
    FACTOR_RANK_LONG_ONLY = "FACTOR_RANK_LONG_ONLY"
    ETF_TREND = "ETF_TREND"
    CUSTOM = "CUSTOM"


class StrategySource(StrEnum):
    BUILTIN = "BUILTIN"
    AGENT_GENERATED = "AGENT_GENERATED"
    HUMAN_AUTHORED = "HUMAN_AUTHORED"


StrategyLifecycleStatus = ApprovalStatus


class FactorLeg(BaseModel):
    factor_id: str
    weight: float = 1.0
    ascending: bool = False
    transform: str | None = None


class PortfolioConstructionSpec(BaseModel):
    method: str = "equal_weight_top_n"
    top_n: int = Field(default=20, gt=0)
    max_single_position_pct: float = Field(default=0.10, gt=0, le=1)
    cash_buffer_pct: float = Field(default=0.02, ge=0, lt=1)
    long_only: bool = True


class RebalanceSpec(BaseModel):
    frequency: str = "daily"
    min_turnover_threshold: float = Field(default=0.0, ge=0, le=1)


class ExecutionAssumptionSpec(BaseModel):
    signal_timing: str = "after_close"
    execution_timing: str = "next_open"
    execution_delay_days: int = Field(default=1, ge=1)
    slippage_bps: float = Field(default=5.0, ge=0)
    cost_model: str = "a_share_default"


class StrategySpec(BaseModel):
    strategy_id: str
    name: str
    version: str = "0.1.0"
    description: str = ""
    kind: StrategyKind = StrategyKind.CUSTOM
    source: StrategySource = StrategySource.AGENT_GENERATED
    universe: str = "stock_etf"
    factors: list[FactorLeg] = Field(default_factory=list)
    portfolio: PortfolioConstructionSpec = Field(default_factory=PortfolioConstructionSpec)
    rebalance: RebalanceSpec = Field(default_factory=RebalanceSpec)
    execution: ExecutionAssumptionSpec = Field(default_factory=ExecutionAssumptionSpec)
    risk_constraints: dict[str, Any] = Field(default_factory=dict)
    tags: list[str] = Field(default_factory=list)


class SavedStrategy(BaseModel):
    strategy_id: str
    name: str
    version: str
    source: StrategySource
    status: ApprovalStatus
    spec: StrategySpec
    implementation_ref: str
    code_path: str | None = None
    tests_path: str | None = None
    report_paths: list[str] = Field(default_factory=list)
    created_at: str = Field(default_factory=shanghai_now_iso)
    updated_at: str = Field(default_factory=shanghai_now_iso)
    created_by: str = "agent"
    approval_file: str | None = None


def strategy_spec_from_agent_spec(data: dict[str, Any]) -> StrategySpec:
    """Convert the legacy agent strategy spec shape into the canonical model."""
    payload = dict(data)
    factors = payload.get("factors") or payload.get("selected_factors") or []
    constraints = payload.get("constraints") or payload.get("risk_constraints") or {}
    constraints = constraints if isinstance(constraints, dict) else {}
    factor_weights = constraints.get("factor_weights")
    factor_weights = factor_weights if isinstance(factor_weights, dict) else {}
    factor_directions = constraints.get("factor_directions")
    factor_directions = factor_directions if isinstance(factor_directions, dict) else {}
    factor_legs: list[FactorLeg] = []
    for item in factors:
        if isinstance(item, str):
            factor_legs.append(
                FactorLeg(
                    factor_id=item,
                    weight=float(factor_weights.get(item, 1.0)),
                    ascending=_direction_is_ascending(factor_directions.get(item, False)),
                )
            )
        elif isinstance(item, dict):
            raw = dict(item)
            if "factor_id" not in raw and "name" in raw:
                raw["factor_id"] = str(raw["name"])
            factor_id = str(raw.get("factor_id", ""))
            if factor_id in factor_weights:
                raw["weight"] = factor_weights[factor_id]
            if factor_id in factor_directions and "ascending" not in raw:
                raw["ascending"] = _direction_is_ascending(factor_directions[factor_id])
            if "direction" in raw and "ascending" not in raw:
                raw["ascending"] = _direction_is_ascending(raw["direction"])
            factor_legs.append(FactorLeg.model_validate(raw))

    portfolio_data = payload.get("portfolio") or payload.get("portfolio_construction") or {}
    if isinstance(portfolio_data, dict) and portfolio_data.get("method") == "equal_weight":
        portfolio_data = {**portfolio_data, "method": "equal_weight_top_n"}
    if not portfolio_data and constraints:
        portfolio_data = {
            key: constraints[key]
            for key in ("top_n", "max_single_position_pct", "cash_buffer_pct", "long_only")
            if key in constraints
        }
        if portfolio_data:
            portfolio_data.setdefault("method", "equal_weight_top_n")

    execution_data = payload.get("execution") or payload.get("execution_assumptions") or {}
    if not execution_data and constraints:
        execution_data = {
            key: constraints[key]
            for key in ("execution_delay_days", "slippage_bps", "cost_model")
            if key in constraints
        }
    if isinstance(execution_data, dict):
        execution_data = _normalize_execution(execution_data)

    kind = payload.get("kind")
    if kind is None:
        idea = f"{payload.get('name', '')} {payload.get('description', '')}".lower()
        kind = (
            StrategyKind.ETF_TREND
            if "etf" in idea and "trend" in idea
            else StrategyKind.FACTOR_RANK_LONG_ONLY
        )

    return StrategySpec(
        strategy_id=str(payload["strategy_id"]),
        name=str(payload.get("name") or payload["strategy_id"]),
        version=str(payload.get("version") or "0.1.0"),
        description=str(payload.get("description") or ""),
        kind=StrategyKind(kind),
        source=StrategySource(payload.get("source") or StrategySource.AGENT_GENERATED),
        universe=str(payload.get("universe") or "stock_etf"),
        factors=factor_legs,
        portfolio=PortfolioConstructionSpec.model_validate(portfolio_data or {}),
        rebalance=RebalanceSpec.model_validate(payload.get("rebalance") or {}),
        execution=ExecutionAssumptionSpec.model_validate(execution_data or {}),
        risk_constraints=dict(payload.get("risk_constraints") or constraints),
        tags=[str(item) for item in payload.get("tags", [])],
    )


def _direction_is_ascending(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {
        "ascending",
        "asc",
        "lower_is_better",
        "low",
        "smaller_is_better",
    }


def _normalize_execution(data: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(data)
    timing = normalized.pop("timing", None)
    if timing is not None:
        normalized.setdefault("execution_timing", timing)
    slippage_model = str(normalized.pop("slippage_model", ""))
    if "5bps" in slippage_model and "slippage_bps" not in normalized:
        normalized["slippage_bps"] = 5.0
    return normalized
