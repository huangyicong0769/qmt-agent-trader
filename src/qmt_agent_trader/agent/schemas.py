"""Structured LLM output schemas."""

from __future__ import annotations

from pydantic import BaseModel


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
