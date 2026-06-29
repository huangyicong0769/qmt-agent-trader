"""Structured schemas for Agent operations.

Existing schemas (HypothesisSpec, ImplementationPlan, ResearchSpec) are preserved.
New schemas for the expanded Agent subsystem are added below.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field

from qmt_agent_trader.agent.permissions import PermissionLevel, ToolCallMode

# ── Original schemas (preserved) ──────────────────────────────────────────────


class HypothesisSpec(BaseModel):
    name: str
    description: str
    intuition: str
    required_data: list[str]
    lookback: int
    universe: list[str]
    expected_behavior: str
    known_risks: list[str]


class ImplementationPlan(BaseModel):
    factor_code_allowed: bool = True
    strategy_code_allowed: bool = True
    live_trading_allowed: bool = False


class ResearchSpec(BaseModel):
    hypothesis: HypothesisSpec
    implementation_plan: ImplementationPlan


# ── New schemas ──────────────────────────────────────────────────────────────


class ToolSpec(BaseModel):
    """Immutable description of a registered tool."""

    name: str
    description: str
    input_schema: dict[str, Any] = Field(default_factory=lambda: {
        "type": "object",
        "properties": {},
        "required": [],
        "additionalProperties": False,
    })
    output_schema: dict[str, Any] = Field(default_factory=dict)
    permission: PermissionLevel = PermissionLevel.READ_ONLY
    side_effect_level: str = "none"  # none | write_generated | write_formal
    timeout_seconds: int = 60
    audit_required: bool = True
    deterministic: bool = False
    llm_callable: bool = True


class ToolContext(BaseModel):
    """Runtime context passed to every tool invocation."""

    run_id: str
    session_id: str | None = None
    experiment_id: str | None = None
    requested_by_llm: bool = True
    call_mode: ToolCallMode | None = None
    dry_run: bool = True
    user_id: str | None = None


# ── Experiment Store schemas ──────────────────────────────────────────────────


class ExperimentStatus(StrEnum):
    CREATED = "CREATED"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    REVIEW_REQUIRED = "REVIEW_REQUIRED"


class ExperimentRecord(BaseModel):
    experiment_id: str
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(tz=UTC)
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(tz=UTC)
    )
    kind: str  # factor_discovery | strategy_engineering | self_bootstrap
    status: ExperimentStatus = ExperimentStatus.CREATED
    hypothesis: dict[str, Any] | None = None
    artifacts: list[str] = Field(default_factory=list)
    metrics: dict[str, Any] = Field(default_factory=dict)
    lessons: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)


# ── Factor / Strategy candidate schemas ───────────────────────────────────────


class FactorSpec(BaseModel):
    """Structured factor specification produced by the Agent."""

    factor_id: str
    name: str
    version: str = "0.1.0"
    inputs: list[str] = Field(default_factory=list)
    lookback: int = 20
    formula: str = ""
    neutralization: dict[str, Any] = Field(default_factory=dict)
    winsorization: dict[str, Any] = Field(default_factory=dict)
    missing_value_policy: str = "drop"
    pit_requirements: dict[str, Any] = Field(default_factory=dict)


class StrategySpec(BaseModel):
    """Structured strategy specification produced by the Agent."""

    strategy_id: str
    name: str
    version: str = "0.1.0"
    universe: str = "stock_etf"
    factors: list[str] = Field(default_factory=list)
    portfolio_construction: dict[str, Any] = Field(default_factory=dict)
    rebalance: dict[str, Any] = Field(default_factory=dict)
    risk_constraints: dict[str, Any] = Field(default_factory=dict)
    execution_assumptions: dict[str, Any] = Field(default_factory=dict)


# ── Sandbox schemas ──────────────────────────────────────────────────────────


class SandboxStaticScanResult(BaseModel):
    status: str  # "PASSED" | "FAILED"
    issues: list[str] = Field(default_factory=list)


class SandboxTestResult(BaseModel):
    status: str  # "PASSED" | "FAILED"
    test_summary: dict[str, Any] = Field(default_factory=dict)
    safety_issues: list[str] = Field(default_factory=list)
