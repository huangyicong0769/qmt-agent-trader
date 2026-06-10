"""FastAPI app and Typer CLI for the Windows gateway."""

from __future__ import annotations

from typing import Any

import typer
import uvicorn
from fastapi import Body, Depends, FastAPI, WebSocket

from qmt_gateway.audit import GatewayAuditLog
from qmt_gateway.auth import NonceStore, require_auth
from qmt_gateway.config import GatewaySettings
from qmt_gateway.qmt_adapter import QMTAdapter
from qmt_gateway.risk import live_gate_reasons, precheck_order_plan
from qmt_gateway.schemas import HealthResponse, OrderPrecheckResponse, SubmitPlanResponse

cli = typer.Typer(help="Windows QMT gateway.")


def create_app(settings: GatewaySettings | None = None) -> FastAPI:
    gateway_settings = settings or GatewaySettings()
    app = FastAPI(title="QMT Gateway", version="0.1.0")
    app.state.settings = gateway_settings
    app.state.nonce_store = NonceStore()
    app.state.adapter = QMTAdapter(xtquant_path=gateway_settings.qmt_xtquant_path)
    app.state.audit = GatewayAuditLog(gateway_settings.audit_jsonl_path)

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse(
            status="ok",
            dry_run=gateway_settings.dry_run,
            live_trading_enabled=gateway_settings.live_trading_enabled,
            allow_order_endpoint=gateway_settings.allow_order_endpoint,
        )

    @app.get("/qmt/status", dependencies=[Depends(require_auth)])
    def qmt_status() -> dict[str, object]:
        return app.state.adapter.status()

    @app.get("/account/asset", dependencies=[Depends(require_auth)])
    def account_asset() -> dict[str, object]:
        return app.state.adapter.asset()

    @app.get("/account/positions", dependencies=[Depends(require_auth)])
    def account_positions() -> dict[str, object]:
        return app.state.adapter.positions()

    @app.get("/account/orders", dependencies=[Depends(require_auth)])
    def account_orders() -> dict[str, object]:
        return app.state.adapter.orders()

    @app.get("/account/trades", dependencies=[Depends(require_auth)])
    def account_trades() -> dict[str, object]:
        return app.state.adapter.trades()

    @app.get("/market/instruments", dependencies=[Depends(require_auth)])
    def market_instruments() -> dict[str, object]:
        return app.state.adapter.instruments()

    @app.get("/market/bars", dependencies=[Depends(require_auth)])
    def market_bars(symbols: str, start: str, end: str, freq: str) -> dict[str, object]:
        return app.state.adapter.bars(symbols=symbols, start=start, end=end, freq=freq)

    @app.get("/market/latest", dependencies=[Depends(require_auth)])
    def market_latest(symbols: str) -> dict[str, object]:
        return app.state.adapter.latest(symbols=symbols)

    @app.post(
        "/orders/precheck",
        response_model=OrderPrecheckResponse,
        dependencies=[Depends(require_auth)],
    )
    def orders_precheck(plan: dict[str, Any] = Body(...)) -> OrderPrecheckResponse:
        reasons = precheck_order_plan(plan)
        app.state.audit.append("orders.precheck", {"accepted": not reasons, "reasons": reasons})
        return OrderPrecheckResponse(accepted=not reasons, reasons=reasons)

    @app.post(
        "/orders/submit-plan",
        response_model=SubmitPlanResponse,
        dependencies=[Depends(require_auth)],
    )
    def orders_submit(plan: dict[str, Any] = Body(...)) -> SubmitPlanResponse:
        reasons = precheck_order_plan(plan)
        reasons.extend(
            live_gate_reasons(
                live_trading_enabled=gateway_settings.live_trading_enabled,
                allow_order_endpoint=gateway_settings.allow_order_endpoint,
            )
        )
        accepted = not reasons
        app.state.audit.append("orders.submit_plan", {"accepted": accepted, "reasons": reasons})
        return SubmitPlanResponse(
            accepted=accepted,
            dry_run=gateway_settings.dry_run,
            idempotency_key=str(plan.get("idempotency_key") or ""),
            execution_id=None,
            reasons=reasons,
        )

    @app.post("/orders/cancel", dependencies=[Depends(require_auth)])
    def orders_cancel(payload: dict[str, Any] = Body(...)) -> dict[str, object]:
        app.state.audit.append(
            "orders.cancel", {"accepted": False, "payload_keys": sorted(payload)}
        )
        return {"accepted": False, "reason": "cancel skeleton only"}

    @app.get("/audit/events", dependencies=[Depends(require_auth)])
    def audit_events() -> dict[str, object]:
        return {"events": app.state.audit.read()}

    @app.websocket("/stream/events")
    async def stream_events(websocket: WebSocket) -> None:
        await websocket.accept()
        await websocket.send_json({"event": "connected", "dry_run": gateway_settings.dry_run})
        await websocket.close()

    return app


app = create_app()


@cli.command("serve")
def serve() -> None:
    settings = GatewaySettings()
    uvicorn.run(
        "qmt_gateway.app:create_app",
        factory=True,
        host=settings.gateway_host,
        port=settings.gateway_port,
    )


@cli.command("health")
def health_cli() -> None:
    settings = GatewaySettings()
    typer.echo(
        {
            "status": "ok",
            "host": settings.gateway_host,
            "port": settings.gateway_port,
            "dry_run": settings.dry_run,
            "live_trading_enabled": settings.live_trading_enabled,
        }
    )


@cli.command("qmt-smoke-test")
def qmt_smoke_test() -> None:
    settings = GatewaySettings()
    adapter = QMTAdapter(xtquant_path=settings.qmt_xtquant_path)
    typer.echo(
        {
            "xtquant_importable": adapter.status()["xtquant_importable"],
            "mini_qmt_path_configured": settings.qmt_miniqmt_path is not None,
            "account_configured": bool(settings.qmt_account_id),
            "real_order_tested": False,
        }
    )
