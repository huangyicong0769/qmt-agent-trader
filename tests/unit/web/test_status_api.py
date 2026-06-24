"""Tests for web status routes."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import SecretStr

from qmt_agent_trader.core.config import Settings
from qmt_agent_trader.web.routes import status


def test_status_api_returns_ok() -> None:
    app = FastAPI()
    app.include_router(status.router)

    response = TestClient(app).get("/")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_config_status_does_not_expose_secret_values(tmp_path, monkeypatch) -> None:
    settings = Settings(
        project_root=tmp_path,
        data_dir=Path("data"),
        log_dir=Path("logs"),
        deepseek_api_key=SecretStr("sk-secret"),
        tushare_token=SecretStr("tushare-secret"),
        qmt_gateway_api_key=SecretStr("gateway-secret"),
        qmt_gateway_hmac_secret=SecretStr("hmac-secret"),
    )
    monkeypatch.setattr(status, "get_settings", lambda: settings)
    app = FastAPI()
    app.include_router(status.router)

    response = TestClient(app).get("/config")

    assert response.status_code == 200
    body = response.text
    assert "sk-secret" not in body
    assert "tushare-secret" not in body
    assert "gateway-secret" not in body
    assert "hmac-secret" not in body
    assert response.json()["deepseek_configured"] is True
