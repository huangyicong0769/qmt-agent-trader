"""Configuration loading for the Mac control plane."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_env: str = "dev"
    project_root: Path = Field(default_factory=lambda: Path.cwd())
    data_dir: Path = Path("data")
    log_dir: Path = Path("logs")
    audit_fsync: bool = True
    audit_rotation_bytes: int | None = None
    cache_ttl_seconds: int = 86400

    tushare_token: SecretStr | None = None
    tushare_quota_profile_source: str = "official_table"
    tushare_points: int | None = 2000
    tushare_max_requests_per_minute: int | None = 200
    tushare_max_requests_per_day_per_api: int | None = 100000

    deepseek_api_key: SecretStr | None = None
    deepseek_base_url: str = "https://api.deepseek.com"
    deepseek_model: str = "deepseek-v4-pro"

    qmt_gateway_base_url: str = "http://192.168.1.100:8765"
    qmt_gateway_api_key: SecretStr | None = None
    qmt_gateway_hmac_secret: SecretStr | None = None

    dry_run: bool = True
    live_trading_enabled: bool = False

    remote_data_max_concurrency: int = 200
    remote_data_min_interval_seconds: float = 0.3
    remote_data_max_days_per_call: int = 366
    remote_data_lock_timeout_seconds: float = 30.0
    remote_data_http_timeout_seconds: float = 300.0
    remote_data_tool_base_timeout_seconds: int = 120
    remote_data_tool_timeout_seconds_per_request: int = 15
    remote_data_tool_max_timeout_seconds: int = 3600
    remote_data_retry_attempts: int = 3
    remote_data_retry_backoff_seconds: float = 2.0

    research_tool_base_timeout_seconds: int = 120
    research_tool_timeout_seconds_per_100k_rows: int = 30
    research_tool_max_timeout_seconds: int = 1800
    backtest_tool_max_timeout_seconds: int = 1800
    factor_eval_tool_max_timeout_seconds: int = 900

    mcp_enabled: bool = False
    mcp_config_path: Path = Path("configs/mcp.servers.json")
    mcp_tool_prefix: str = "mcp"
    mcp_default_timeout_seconds: int = 60

    @property
    def resolved_data_dir(self) -> Path:
        return self.project_root / self.data_dir

    @property
    def resolved_log_dir(self) -> Path:
        return self.project_root / self.log_dir


@lru_cache
def get_settings() -> Settings:
    return Settings()
