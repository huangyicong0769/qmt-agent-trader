"""Tests for artifact path safety."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

from qmt_agent_trader.web.config import WebConfig
from qmt_agent_trader.web.routes import artifacts
from qmt_agent_trader.web.routes.artifacts import encode_artifact_id


def test_artifact_path_safety_allows_configured_roots(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    allowed = tmp_path / "data" / "reports" / "report.md"
    allowed.parent.mkdir(parents=True)
    allowed.write_text("ok", encoding="utf-8")

    assert WebConfig().is_path_safe(allowed)


def test_artifact_path_safety_blocks_traversal(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    secret = tmp_path / "outside.txt"
    secret.write_text("nope", encoding="utf-8")

    assert not WebConfig().is_path_safe(secret)
    assert not WebConfig().is_path_safe(Path("../outside.txt"))


def test_artifact_content_route_rejects_unsafe_path(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    unsafe = tmp_path / "outside.txt"
    unsafe.write_text("nope", encoding="utf-8")
    monkeypatch.setattr(artifacts, "get_web_config", lambda: WebConfig())
    app = FastAPI()
    app.include_router(artifacts.router)

    response = TestClient(app).get(f"/{encode_artifact_id(unsafe)}/content")

    assert response.status_code == 404
