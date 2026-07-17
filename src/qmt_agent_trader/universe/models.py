"""Declarative universe models.

Universe specs are research artifacts. They intentionally do not accept
arbitrary expressions or executable selection code.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from qmt_agent_trader.core.ids import shanghai_now_iso
from qmt_agent_trader.universe.validators import (
    ALLOWED_RULE_OPERATORS,
    looks_like_code_expression,
    normalize_symbols,
)

UniverseSource = Literal["builtin", "agent_generated", "user_defined", "imported"]
AssetType = Literal["stock", "etf"]
UniverseMode = Literal["snapshot", "rolling"]
RebalanceFrequency = Literal["daily", "weekly", "monthly"]
SelectionMode = Literal[
    "all",
    "explicit_symbols",
    "industry",
    "theme",
    "index_constituents",
    "etf_category",
    "fundamental_filter",
    "liquidity_filter",
    "composite",
]
RuleOperator = Literal[
    "eq",
    "ne",
    "in",
    "not_in",
    "gt",
    "gte",
    "lt",
    "lte",
    "between",
    "contains",
    "starts_with",
    "ends_with",
]


class UniverseRule(BaseModel):
    model_config = ConfigDict(extra="forbid")

    field: str
    operator: RuleOperator
    value: Any

    @field_validator("field")
    @classmethod
    def _field_must_be_declarative(cls, value: str) -> str:
        text = value.strip()
        if not text:
            raise ValueError("rule field cannot be empty")
        if looks_like_code_expression(text):
            raise ValueError("rule field cannot contain executable expressions")
        return text

    @field_validator("operator", mode="before")
    @classmethod
    def _operator_must_be_supported(cls, value: Any) -> Any:
        if str(value) not in ALLOWED_RULE_OPERATORS:
            raise ValueError(f"unsupported rule operator: {value}")
        return value

    @field_validator("value")
    @classmethod
    def _value_must_not_be_code(cls, value: Any) -> Any:
        if looks_like_code_expression(value):
            raise ValueError("rule value cannot contain executable expressions")
        if isinstance(value, list):
            for item in value:
                if looks_like_code_expression(item):
                    raise ValueError("rule value cannot contain executable expressions")
        return value


class UniverseSelection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mode: SelectionMode
    symbols: list[str] = Field(default_factory=list)
    industries: list[str] = Field(default_factory=list)
    theme_concepts: list[str] = Field(default_factory=list)
    index_codes: list[str] = Field(default_factory=list)
    rules: list[UniverseRule] = Field(default_factory=list)

    @field_validator("symbols", mode="before")
    @classmethod
    def _normalize_symbols(cls, value: Any) -> Any:
        if value is None:
            return []
        if not isinstance(value, list):
            raise ValueError("symbols must be a list")
        return normalize_symbols(value)

    @field_validator("industries", "theme_concepts", "index_codes", mode="before")
    @classmethod
    def _normalize_string_list(cls, value: Any) -> list[str]:
        if value is None:
            return []
        if not isinstance(value, list):
            raise ValueError("value must be a list")
        result: list[str] = []
        for item in value:
            text = str(item).strip()
            if text and text not in result:
                result.append(text)
        return result

    @model_validator(mode="after")
    def _required_selection_values_present(self) -> UniverseSelection:
        if self.mode == "explicit_symbols" and not self.symbols:
            raise ValueError("explicit_symbols selection requires symbols")
        if self.mode == "industry" and not self.industries:
            raise ValueError("industry selection requires industries")
        if self.mode == "theme" and not self.theme_concepts:
            raise ValueError("theme selection requires theme_concepts")
        if self.mode == "index_constituents" and not self.index_codes:
            raise ValueError("index_constituents selection requires index_codes")
        if self.mode == "etf_category" and not self.theme_concepts:
            raise ValueError("etf_category selection requires categories")
        return self


class UniverseFilters(BaseModel):
    model_config = ConfigDict(extra="forbid")

    exclude_st: bool = True
    exclude_suspended: bool = True
    min_listed_days: int = Field(default=60, ge=0)
    min_avg_amount_20d: float | None = Field(default=None, ge=0)
    min_avg_volume_20d: float | None = Field(default=None, ge=0)
    min_market_cap: float | None = Field(default=None, ge=0)
    max_market_cap: float | None = Field(default=None, ge=0)
    require_bar_coverage: bool = True
    require_fundamental_coverage: bool = False

    @model_validator(mode="after")
    def _market_cap_bounds_are_ordered(self) -> UniverseFilters:
        if (
            self.min_market_cap is not None
            and self.max_market_cap is not None
            and self.min_market_cap > self.max_market_cap
        ):
            raise ValueError("min_market_cap cannot exceed max_market_cap")
        return self


class UniverseRanking(BaseModel):
    model_config = ConfigDict(extra="forbid")

    field: str
    ascending: bool = False
    top_n: int | None = Field(default=None, gt=0)

    @field_validator("field")
    @classmethod
    def _field_must_be_declarative(cls, value: str) -> str:
        text = value.strip()
        if not text:
            raise ValueError("ranking field cannot be empty")
        if looks_like_code_expression(text):
            raise ValueError("ranking field cannot contain executable expressions")
        return text


class UniverseSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    universe_id: str
    name: str
    description: str = ""
    version: str = "0.1.0"
    source: UniverseSource
    asset_types: list[AssetType]
    selection: UniverseSelection
    filters: UniverseFilters = Field(default_factory=UniverseFilters)
    ranking: UniverseRanking | None = None
    max_symbols: int | None = Field(default=None, gt=0)
    mode: UniverseMode = "snapshot"
    rebalance_frequency: RebalanceFrequency = "daily"
    created_by: str = "agent"
    created_at: str = Field(default_factory=shanghai_now_iso)
    research_only: bool = True
    live_trading_allowed: bool = False
    approval_required: bool = True
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("universe_id", "name")
    @classmethod
    def _required_text(cls, value: str) -> str:
        text = value.strip()
        if not text:
            raise ValueError("value cannot be empty")
        return text

    @field_validator("asset_types", mode="before")
    @classmethod
    def _asset_types_must_be_unique(cls, value: Any) -> Any:
        if value is None:
            raise ValueError("asset_types cannot be empty")
        if not isinstance(value, list):
            raise ValueError("asset_types must be a list")
        result: list[Any] = []
        for item in value:
            if item not in result:
                result.append(item)
        if not result:
            raise ValueError("asset_types cannot be empty")
        return result

    @model_validator(mode="after")
    def _agent_generated_is_research_only(self) -> UniverseSpec:
        if self.source == "agent_generated":
            self.research_only = True
            self.live_trading_allowed = False
            self.approval_required = True
        return self
