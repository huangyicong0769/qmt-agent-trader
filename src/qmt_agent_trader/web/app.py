"""FastAPI application factory for QMT Agent Studio."""

from __future__ import annotations

from fastapi import FastAPI
from nicegui import ui

from qmt_agent_trader.web.routes import (
    artifacts,
    audit,
    chat,
    experiments,
    status,
    tools,
    workflows,
)
from qmt_agent_trader.web.ui.main import create_ui


def create_app() -> FastAPI:
    app = FastAPI(title="QMT Agent Studio", version="0.1.0")

    app.include_router(chat.router, prefix="/api/chat", tags=["chat"])
    app.include_router(tools.router, prefix="/api/tools", tags=["tools"])
    app.include_router(workflows.router, prefix="/api/workflows", tags=["workflows"])
    app.include_router(experiments.router, prefix="/api/experiments", tags=["experiments"])
    app.include_router(artifacts.router, prefix="/api/artifacts", tags=["artifacts"])
    app.include_router(audit.router, prefix="/api/audit", tags=["audit"])
    app.include_router(status.router, prefix="/api/status", tags=["status"])

    # NiceGUI mounts a catch-all route at "/"; exact aliases keep spec paths
    # like "/api/status" working without relying on slash redirects.
    app.add_api_route("/api/tools", tools.list_tools, methods=["GET"], include_in_schema=False)
    app.add_api_route(
        "/api/experiments",
        experiments.list_experiments,
        methods=["GET"],
        include_in_schema=False,
    )
    app.add_api_route(
        "/api/artifacts",
        artifacts.list_artifacts,
        methods=["GET"],
        include_in_schema=False,
    )
    app.add_api_route("/api/status", status.get_status, methods=["GET"], include_in_schema=False)

    create_ui()
    ui.run_with(app, mount_path="/", title="QMT Agent Studio")

    return app
