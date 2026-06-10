"""Gateway settings."""

from __future__ import annotations

from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class GatewaySettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    gateway_host: str = "0.0.0.0"
    gateway_port: int = 8765

    qmt_xtquant_path: Path | None = None
    qmt_miniqmt_path: Path | None = None
    qmt_account_id: str | None = None
    qmt_account_type: str = "STOCK"
    qmt_session_id: int = 123456

    gateway_api_key: str = ""
    gateway_hmac_secret: str = ""

    dry_run: bool = True
    live_trading_enabled: bool = False
    allow_order_endpoint: bool = False
    allowed_client_cidrs: list[str] = Field(default_factory=lambda: ["192.168.1.0/24"])

    audit_jsonl_path: Path = Path("logs/gateway_audit.jsonl")

    @field_validator("allowed_client_cidrs", mode="before")
    @classmethod
    def parse_cidrs(cls, value: object) -> list[str]:
        if isinstance(value, str):
            return [item.strip() for item in value.split(",") if item.strip()]
        if isinstance(value, list):
            return [str(item) for item in value]
        return ["192.168.1.0/24"]
