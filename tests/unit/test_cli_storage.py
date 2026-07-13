from __future__ import annotations

import json

from typer.testing import CliRunner

from qmt_agent_trader.cli.main import app
from qmt_agent_trader.core.config import Settings
from qmt_agent_trader.persistence.errors import StorageUnavailableError


def test_storage_help_lists_all_operations() -> None:
    result = CliRunner().invoke(app, ["storage", "--help"])
    assert result.exit_code == 0
    for command in ("inventory", "verify", "migrate", "backup", "locks", "quarantine", "reset"):
        assert command in result.stdout


def test_storage_inventory_and_verify_exit_codes(monkeypatch, tmp_path) -> None:
    settings = Settings(project_root=tmp_path)
    monkeypatch.setattr("qmt_agent_trader.cli.main._settings", lambda: settings)
    runner = CliRunner()
    inventory = runner.invoke(app, ["storage", "inventory"])
    assert inventory.exit_code == 0
    assert any(item["name"] == "control_db" for item in json.loads(inventory.stdout))

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


def test_storage_reset_requires_dry_run_digest_and_executes(monkeypatch, tmp_path) -> None:
    settings = Settings(project_root=tmp_path)
    monkeypatch.setattr("qmt_agent_trader.cli.main._settings", lambda: settings)
    stale = tmp_path / "sessions/legacy.json"
    stale.parent.mkdir(parents=True)
    stale.write_text("legacy")
    runner = CliRunner()

    planned = runner.invoke(
        app, ["storage", "reset", "--profile", "preserve-raw", "--dry-run"]
    )
    assert planned.exit_code == 0
    plan = json.loads(planned.stdout)
    assert plan["status"] == "planned"
    assert stale.exists()

    missing = runner.invoke(app, ["storage", "reset", "--profile", "preserve-raw"])
    assert missing.exit_code == 1
    assert json.loads(missing.stdout)["status"] == "rejected"

    completed = runner.invoke(
        app,
        [
            "storage",
            "reset",
            "--profile",
            "preserve-raw",
            "--confirm",
            plan["digest"],
        ],
    )
    assert completed.exit_code == 0
    assert json.loads(completed.stdout)["status"] == "completed"
    assert not stale.exists()


def test_every_storage_command_happy_path(monkeypatch, tmp_path) -> None:
    settings = Settings(project_root=tmp_path)
    monkeypatch.setattr("qmt_agent_trader.cli.main._settings", lambda: settings)
    runner = CliRunner()

    assert runner.invoke(app, ["storage", "verify"]).exit_code == 0
    first = runner.invoke(app, ["storage", "migrate"])
    second = runner.invoke(app, ["storage", "migrate"])
    assert first.exit_code == second.exit_code == 0
    assert json.loads(second.stdout)["migrations"] == []
    backup = runner.invoke(app, ["storage", "backup"])
    assert backup.exit_code == 0
    assert json.loads(backup.stdout)["status"] == "ok"
    assert runner.invoke(app, ["storage", "locks"]).exit_code == 0

    broken = tmp_path / "sessions/bad.json"
    broken.parent.mkdir(parents=True, exist_ok=True)
    broken.write_text("{broken")
    quarantined = runner.invoke(app, ["storage", "quarantine", "sessions", "bad.json"])
    assert quarantined.exit_code == 0
    assert json.loads(quarantined.stdout)["status"] == "quarantined"


def test_every_storage_command_has_structured_failure_exit(monkeypatch) -> None:
    failure = StorageUnavailableError(
        store_name="test",
        operation="test",
        reason="unavailable",
    )

    class FailingOperations:
        def __getattr__(self, _name: str):
            def fail(*_args: object, **_kwargs: object) -> object:
                raise failure

            return fail

    monkeypatch.setattr("qmt_agent_trader.cli.main._storage_operations", FailingOperations)
    runner = CliRunner()
    commands = [
        ["inventory"],
        ["verify"],
        ["migrate"],
        ["backup"],
        ["locks"],
        ["quarantine", "sessions", "bad.json"],
        ["reset", "--profile", "preserve-raw", "--dry-run"],
    ]
    for command in commands:
        result = runner.invoke(app, ["storage", *command])
        assert result.exit_code == 1, command
        assert json.loads(result.stdout)["error_type"] == "StorageUnavailableError"
