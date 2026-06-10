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

    tushare_token: SecretStr | None = None

    deepseek_api_key: SecretStr | None = None
    deepseek_base_url: str = "https://api.deepseek.com"
    deepseek_model: str = "deepseek-v4-pro"

    qmt_gateway_base_url: str = "http://192.168.1.100:8765"
    qmt_gateway_api_key: SecretStr | None = None
    qmt_gateway_hmac_secret: SecretStr | None = None

    dry_run: bool = True
    live_trading_enabled: bool = False

    @property
    def resolved_data_dir(self) -> Path:
        return self.project_root / self.data_dir

    @property
    def resolved_log_dir(self) -> Path:
        return self.project_root / self.log_dir


@lru_cache
def get_settings() -> Settings:
    return Settings()
