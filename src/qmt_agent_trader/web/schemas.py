"""Pydantic models for the Agent Studio web API."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

from qmt_agent_trader.core.ids import new_id, shanghai_now_iso


class ChatMessage(BaseModel):
    message_id: str = Field(default_factory=lambda: new_id("msg"))
    session_id: str
    role: Literal["user", "assistant", "system", "info", "tool", "done", "error"]
    content: str
    created_at: str = Field(default_factory=shanghai_now_iso)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ChatSession(BaseModel):
    schema_version: Literal[2] = 2
    revision: int = Field(default=0, ge=0)
    session_id: str = Field(default_factory=lambda: new_id("chat"))
    title: str = "New research chat"
    created_at: str = Field(default_factory=shanghai_now_iso)
    updated_at: str = Field(default_factory=shanghai_now_iso)
    messages: list[ChatMessage] = Field(default_factory=list)
    context: dict[str, Any] = Field(default_factory=dict)


class AdvancedOptions(BaseModel):
    universe: str = "auto"
    start_date: str | None = None
    end_date: str | None = None
    max_hypotheses: int | None = None
    risk_profile: str | None = None
    budget_mode: str = "balanced"


class CreateChatSessionRequest(BaseModel):
    title: str | None = None
    context: dict[str, Any] = Field(default_factory=dict)


class SendMessageRequest(BaseModel):
    content: str
    advanced: AdvancedOptions | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class StartChatRunRequest(BaseModel):
    message: str | None = None
    content: str | None = None
    interrupt: bool = False


class ToolRunRequest(BaseModel):
    input_data: dict[str, Any] = Field(default_factory=dict)
    experiment_id: str | None = None
    dry_run: bool = True
    user_id: str | None = "web"


class ToolRunResponse(BaseModel):
    run_id: str
    tool_name: str
    status: Literal["ok", "permission_denied", "error"]
    result: dict[str, Any] = Field(default_factory=dict)
    error_message: str | None = None


class WorkflowRunRequest(BaseModel):
    theme: str | None = None
    strategy_idea: str | None = None
    selected_factors: list[str] = Field(default_factory=list)
    universe: str = "stock_etf"
    start_date: str = "20200101"
    end_date: str = "20260624"
    recent_experiment_ids: list[str] = Field(default_factory=list)


class WorkflowRunResponse(BaseModel):
    run_id: str
    workflow_type: str
    status: str
    experiment_id: str | None = None
    message: str
    result: dict[str, Any] = Field(default_factory=dict)


class StatusResponse(BaseModel):
    status: str
    service: str
    version: str
    time: str


class ConfigStatusResponse(BaseModel):
    app_env: str
    project_root: str
    data_dir: str
    log_dir: str
    dry_run: bool
    live_trading_enabled: bool
    deepseek_configured: bool
    tushare_configured: bool
    qmt_gateway_base_url: str
    qmt_gateway_api_key_configured: bool
    qmt_gateway_hmac_configured: bool


class DataStatusResponse(BaseModel):
    data_dir: str
    log_dir: str
    data_dir_exists: bool
    log_dir_exists: bool
    experiment_count: int
    audit_log_count: int


class ExperimentSummary(BaseModel):
    experiment_id: str
    kind: str
    status: str
    created_at: datetime
    updated_at: datetime
    tags: list[str] = Field(default_factory=list)
    artifact_count: int = 0


class ExperimentDetail(ExperimentSummary):
    hypothesis: dict[str, Any] | None = None
    artifacts: list[str] = Field(default_factory=list)
    metrics: dict[str, Any] = Field(default_factory=dict)
    lessons: list[str] = Field(default_factory=list)


class ArtifactSummary(BaseModel):
    artifact_id: str
    name: str
    path: str
    size_bytes: int
    modified_at: datetime


class ArtifactDetail(BaseModel):
    artifact: ArtifactSummary
    content: str


class AuditSummary(BaseModel):
    timestamp: str
    run_id: str
    experiment_id: str | None = None
    tool_name: str
    permission: str
    requested_by_llm: bool
    input_hash: str
    output_hash: str
    status: str
    error_message: str | None = None
    duration_ms: int
