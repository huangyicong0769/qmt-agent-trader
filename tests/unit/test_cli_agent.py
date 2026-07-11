from __future__ import annotations

import json
from types import SimpleNamespace

import pandas as pd
from typer.testing import CliRunner

from qmt_agent_trader.cli.main import app
from qmt_agent_trader.core.config import Settings
from qmt_agent_trader.core.types import ApprovalStatus
from qmt_agent_trader.data.providers.tushare.quota import new_usage_record
from qmt_agent_trader.data.storage import DataLake
from qmt_agent_trader.strategy.approval import read_approval_file
from qmt_agent_trader.strategy.models import SavedStrategy, StrategySource, StrategySpec


def test_agent_call_tool_uses_agent_tool_registry_for_query_bars() -> None:
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "agent",
            "call-tool",
            "--name",
            "query_bars",
            "--params",
            json.dumps(
                {
                    "symbol": "000001.SZ",
                    "start_date": "20260101",
                    "end_date": "20260105",
                }
            ),
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["tool"] == "query_bars"
    assert "metadata" in payload["result"]


def test_agent_ask_json_preserves_legacy_output_shape(monkeypatch) -> None:
    class FakeRuntime:
        def ask(
            self,
            prompt: str,
            *,
            max_rounds: int = 100,
            session_id: str | None = None,
        ):
            assert prompt == "你好"
            assert max_rounds == 3
            assert session_id == "chat_cli"
            return SimpleNamespace(
                content="回答",
                tool_calls=[
                    SimpleNamespace(
                        name="todo_get_status",
                        arguments={},
                        result={"status": "ok"},
                    )
                ],
            )

    monkeypatch.setattr(
        "qmt_agent_trader.cli.main.build_default_runtime",
        lambda _settings: FakeRuntime(),
    )
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "agent",
            "ask",
            "--prompt",
            "你好",
            "--max-rounds",
            "3",
            "--session-id",
            "chat_cli",
            "--json",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload == {
        "content": "回答",
        "tool_calls": [
            {
                "name": "todo_get_status",
                "arguments": {},
                "result": {"status": "ok"},
            }
        ],
    }


def test_strategy_approve_rejects_non_review_required_candidate(monkeypatch) -> None:
    saved = SavedStrategy(
        strategy_id="strat_draft",
        name="draft",
        version="0.1.0",
        source=StrategySource.AGENT_GENERATED,
        status=ApprovalStatus.GENERATED_BY_LLM,
        spec=StrategySpec(strategy_id="strat_draft", name="draft"),
        implementation_ref="file:strategy.py",
    )

    class FakeRegistry:
        def get_strategy(self, strategy_id: str) -> SavedStrategy | None:
            return saved if strategy_id == "strat_draft" else None

    monkeypatch.setattr("qmt_agent_trader.cli.main._strategy_registry", lambda: FakeRegistry())

    result = CliRunner().invoke(app, ["strategy", "approve", "--strategy-id", "strat_draft"])

    assert result.exit_code != 0
    assert "REVIEW_REQUIRED" in result.output


def test_strategy_approve_resumes_after_registry_attach_failure(monkeypatch, tmp_path) -> None:
    saved = SavedStrategy(
        strategy_id="strat_resume",
        name="resume",
        version="0.1.0",
        source=StrategySource.AGENT_GENERATED,
        status=ApprovalStatus.REVIEW_REQUIRED,
        spec=StrategySpec(strategy_id="strat_resume", name="resume"),
        implementation_ref="file:strategy.py",
    )

    class FakeRegistry:
        attach_calls = 0
        status_updates = 0

        def get_strategy(self, strategy_id: str) -> SavedStrategy | None:
            return saved if strategy_id == saved.strategy_id else None

        def attach_approval(self, strategy_id: str, approval_file: str) -> SavedStrategy:
            self.attach_calls += 1
            if self.attach_calls == 1:
                raise RuntimeError("injected registry attach failure")
            return saved

        def update_status(self, strategy_id: str, status, *, trusted: bool = False):
            assert trusted is True
            self.status_updates += 1
            return saved

    registry = FakeRegistry()
    settings = Settings(
        project_root=tmp_path,
        qmt_gateway_api_key=None,
        qmt_gateway_hmac_secret=None,
        deepseek_api_key=None,
    )
    monkeypatch.setattr("qmt_agent_trader.cli.main._strategy_registry", lambda: registry)
    monkeypatch.setattr("qmt_agent_trader.cli.main._settings", lambda: settings)

    runner = CliRunner()
    first = runner.invoke(app, ["strategy", "approve", "--strategy-id", "strat_resume"])
    approval_path = tmp_path / "approvals/strat_resume_0.1.0.approval.yaml"
    first_approval = read_approval_file(approval_path)
    second = runner.invoke(app, ["strategy", "approve", "--strategy-id", "strat_resume"])
    resumed_approval = read_approval_file(approval_path)

    assert first.exit_code != 0
    assert second.exit_code == 0
    assert registry.attach_calls == 2 and registry.status_updates == 1
    assert resumed_approval.approved_at == first_approval.approved_at
    assert resumed_approval.approved_by == "human"


def test_repair_tushare_ledger_is_read_only_without_explicit_quarantine(
    monkeypatch,
    tmp_path,
) -> None:
    lake = DataLake(root=tmp_path / "lake", duckdb_path=tmp_path / "db.duckdb")
    legacy = lake.root / "metadata" / "tushare_usage_ledger.parquet"
    legacy.parent.mkdir(parents=True, exist_ok=True)
    legacy.write_bytes(b"PAR1broken-ledger-pagePAR1")
    monkeypatch.setattr("qmt_agent_trader.cli.main._data_lake", lambda: lake)

    result = CliRunner().invoke(app, ["data", "repair-tushare-ledger"])

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["status"] == "CORRUPT"
    assert payload["modified"] is False
    assert legacy.exists()


def test_repair_tushare_ledger_explicitly_quarantines_and_records_reset(
    monkeypatch,
    tmp_path,
) -> None:
    lake = DataLake(root=tmp_path / "lake", duckdb_path=tmp_path / "db.duckdb")
    legacy = lake.root / "metadata" / "tushare_usage_ledger.parquet"
    legacy.parent.mkdir(parents=True, exist_ok=True)
    legacy.write_bytes(b"PAR1broken-ledger-pagePAR1")
    monkeypatch.setattr("qmt_agent_trader.cli.main._data_lake", lambda: lake)

    result = CliRunner().invoke(
        app,
        ["data", "repair-tushare-ledger", "--quarantine-corrupt"],
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["status"] == "QUARANTINED"
    assert payload["history_reset"] is True
    assert not legacy.exists()


def test_data_validate_recommended_storage_command_exists_and_returns_output(
    monkeypatch, tmp_path
) -> None:
    lake = DataLake(root=tmp_path / "lake", duckdb_path=tmp_path / "db.duckdb")
    monkeypatch.setattr("qmt_agent_trader.cli.main._data_lake", lambda: lake)

    result = CliRunner().invoke(app, ["data", "validate"])

    assert result.exit_code == 0
    assert '"status": "missing_data"' in result.stdout
    assert '"duckdb_exists": false' in result.stdout


def test_data_plan_fetch_migrates_healthy_legacy_usage_ledger(
    monkeypatch,
    tmp_path,
) -> None:
    lake = DataLake(root=tmp_path / "lake", duckdb_path=tmp_path / "db.duckdb")
    legacy = lake.root / "metadata" / "tushare_usage_ledger.parquet"
    legacy.parent.mkdir(parents=True, exist_ok=True)
    record = new_usage_record(
        api_name="daily_basic",
        params={"trade_date": "20240102"},
        fields=["ts_code", "trade_date"],
        status="SUCCESS",
        execution_mode="manual",
    ).model_dump(mode="json")
    record["params_redacted"] = "{}"
    record["fields"] = '["ts_code", "trade_date"]'
    pd.DataFrame([record]).to_parquet(legacy, index=False)
    monkeypatch.setattr("qmt_agent_trader.cli.main._data_lake", lambda: lake)

    result = CliRunner().invoke(
        app,
        [
            "data",
            "plan-fetch",
            "--api",
            "daily_basic",
            "--symbols",
            "000001.SZ",
            "--fields",
            "ts_code,trade_date,pe_ttm",
            "--from",
            "20240101",
            "--to",
            "20240131",
        ],
    )

    assert result.exit_code == 0
    assert not legacy.exists()
    assert len(list((legacy.parent / "archive").glob("*.parquet"))) == 1
