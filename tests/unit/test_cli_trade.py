from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from qmt_agent_trader.broker.order_plan import OrderPlan
from qmt_agent_trader.cli.main import _artifact_store, app
from qmt_agent_trader.core.config import Settings
from qmt_agent_trader.persistence.paths import PersistencePaths
from qmt_agent_trader.services.order_plan_service import (
    build_sample_paper_order_plan,
    save_order_plan,
)

runner = CliRunner()


def _configure_project(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Settings:
    settings = Settings(project_root=tmp_path)
    monkeypatch.setattr("qmt_agent_trader.cli.main._settings", lambda: settings)
    return settings


def _fail_if_called(*_args: object, **_kwargs: object) -> None:
    pytest.fail("trade side effect occurred after plan load failure")


def test_trade_risk_check_reports_missing_plan_as_bad_parameter(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _configure_project(monkeypatch, tmp_path)
    monkeypatch.setattr("qmt_agent_trader.cli.main.run_order_plan_risk_checks", _fail_if_called)
    monkeypatch.setattr("qmt_agent_trader.cli.main.append_order_plan_event", _fail_if_called)
    monkeypatch.setattr("qmt_agent_trader.cli.main._audit_logger", _fail_if_called)

    result = runner.invoke(app, ["trade", "risk-check", "--plan", "missing"])

    assert result.exit_code != 0
    assert "missing" in result.output.lower()
    assert "traceback" not in result.output.lower()


def test_trade_risk_check_reports_tampered_plan_without_side_effects(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    settings = _configure_project(monkeypatch, tmp_path)
    paths = PersistencePaths.from_settings(settings)
    plan: OrderPlan = build_sample_paper_order_plan("s1")

    path = save_order_plan(
        plan,
        artifact_store=_artifact_store(paths.order_plans_root),
    )
    path.write_text(
        path.read_text(encoding="utf-8").replace("paper_account", "tampered_account"),
        encoding="utf-8",
    )
    monkeypatch.setattr("qmt_agent_trader.cli.main.run_order_plan_risk_checks", _fail_if_called)
    monkeypatch.setattr("qmt_agent_trader.cli.main.append_order_plan_event", _fail_if_called)
    monkeypatch.setattr("qmt_agent_trader.cli.main._audit_logger", _fail_if_called)

    result = runner.invoke(
        app,
        ["trade", "risk-check", "--plan", plan.order_plan_id],
    )

    assert result.exit_code != 0
    assert "hash_mismatch" in result.output.lower()
    assert "traceback" not in result.output.lower()
