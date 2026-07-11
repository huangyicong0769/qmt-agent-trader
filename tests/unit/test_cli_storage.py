from __future__ import annotations

import json

from typer.testing import CliRunner

from qmt_agent_trader.cli.main import app
from qmt_agent_trader.core.config import Settings


def test_storage_help_lists_all_operations() -> None:
    result = CliRunner().invoke(app, ["storage", "--help"])
    assert result.exit_code == 0
    for command in ("inventory", "verify", "migrate", "backup", "locks", "quarantine"):
        assert command in result.stdout


def test_storage_inventory_and_verify_exit_codes(monkeypatch, tmp_path) -> None:
    settings = Settings(project_root=tmp_path)
    monkeypatch.setattr("qmt_agent_trader.cli.main._settings", lambda: settings)
    runner = CliRunner()
    inventory = runner.invoke(app, ["storage", "inventory"])
    assert inventory.exit_code == 0
    assert any(item["name"] == "control_db_path" for item in json.loads(inventory.stdout))

    broken = tmp_path / "data/lake/raw/broken.parquet"
    broken.parent.mkdir(parents=True)
    broken.write_bytes(b"bad")
    verify = runner.invoke(app, ["storage", "verify", "--deep"])
    assert verify.exit_code == 1
    assert json.loads(verify.stdout)["healthy"] is False


def test_storage_migrate_dry_run_has_no_mutation(monkeypatch, tmp_path) -> None:
    settings = Settings(project_root=tmp_path)
    monkeypatch.setattr("qmt_agent_trader.cli.main._settings", lambda: settings)
    before = list(tmp_path.rglob("*"))
    result = CliRunner().invoke(app, ["storage", "migrate", "--dry-run"])
    assert result.exit_code == 0
    assert json.loads(result.stdout)["dry_run"] is True
    assert list(tmp_path.rglob("*")) == before
