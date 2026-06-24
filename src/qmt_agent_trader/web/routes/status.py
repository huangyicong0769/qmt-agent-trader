"""Status and safe configuration API routes."""

from __future__ import annotations

from fastapi import APIRouter

from qmt_agent_trader import __version__
from qmt_agent_trader.core.config import get_settings
from qmt_agent_trader.core.ids import shanghai_now_iso
from qmt_agent_trader.web.schemas import (
    ConfigStatusResponse,
    DataStatusResponse,
    StatusResponse,
)

router = APIRouter()


@router.get("/", response_model=StatusResponse)
async def get_status() -> StatusResponse:
    return StatusResponse(
        status="ok",
        service="QMT Agent Studio",
        version=__version__,
        time=shanghai_now_iso(),
    )


@router.get("/config", response_model=ConfigStatusResponse)
async def get_config_status() -> ConfigStatusResponse:
    settings = get_settings()
    return ConfigStatusResponse(
        app_env=settings.app_env,
        project_root=str(settings.project_root),
        data_dir=str(settings.data_dir),
        log_dir=str(settings.log_dir),
        dry_run=settings.dry_run,
        live_trading_enabled=settings.live_trading_enabled,
        deepseek_configured=settings.deepseek_api_key is not None,
        tushare_configured=settings.tushare_token is not None,
        qmt_gateway_base_url=settings.qmt_gateway_base_url,
        qmt_gateway_api_key_configured=settings.qmt_gateway_api_key is not None,
        qmt_gateway_hmac_configured=settings.qmt_gateway_hmac_secret is not None,
    )


@router.get("/data", response_model=DataStatusResponse)
async def get_data_status() -> DataStatusResponse:
    settings = get_settings()
    experiments_dir = settings.resolved_data_dir / "experiments"
    audit_dir = settings.resolved_log_dir / "audit"
    return DataStatusResponse(
        data_dir=str(settings.resolved_data_dir),
        log_dir=str(settings.resolved_log_dir),
        data_dir_exists=settings.resolved_data_dir.exists(),
        log_dir_exists=settings.resolved_log_dir.exists(),
        experiment_count=(
            len(list(experiments_dir.glob("exp_*.json"))) if experiments_dir.exists() else 0
        ),
        audit_log_count=len(list(audit_dir.glob("*.jsonl"))) if audit_dir.exists() else 0,
    )
