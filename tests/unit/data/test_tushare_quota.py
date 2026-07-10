from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from multiprocessing import get_context
from pathlib import Path

import duckdb
import pandas as pd
import pyarrow.parquet as pq
import pytest

from qmt_agent_trader.data.providers.tushare import ledger_migration
from qmt_agent_trader.data.providers.tushare import quota as quota_module
from qmt_agent_trader.data.providers.tushare.ledger_migration import (
    repair_tushare_usage_ledger,
)
from qmt_agent_trader.data.providers.tushare.quota import (
    DEFAULT_TUSHARE_2000_POINT_PROFILE,
    TushareAccountQuotaProfile,
    TushareFetchCostEstimate,
    TushareQuotaManager,
    TushareUsageLedger,
    TushareUsageLedgerCorruptError,
    new_usage_record,
    normalized_request_hash,
)
from qmt_agent_trader.data.storage import DataLake


def test_default_tushare_quota_profile_is_2000_point_account() -> None:
    profile = DEFAULT_TUSHARE_2000_POINT_PROFILE

    assert profile.source == "official_table"
    assert profile.points == 2000
    assert profile.tier == "2000+"
    assert profile.max_requests_per_minute == 200
    assert profile.max_requests_per_day_per_api == 100000
    assert profile.minute_quota_scope == "account_global"
    assert profile.daily_quota_scope == "per_api"


def test_quota_manager_approves_116_requests_with_zero_usage() -> None:
    manager = TushareQuotaManager(profile=DEFAULT_TUSHARE_2000_POINT_PROFILE)
    state = manager.current_state()
    cost = _cost("daily_basic", 116)

    decision = manager.evaluate(cost, state, execution_mode="autonomous")

    assert decision.status == "APPROVED_BY_ACCOUNT_QUOTA"
    assert decision.safe_to_execute_now is True


def test_quota_manager_recommends_rate_pacing_when_minute_remainder_is_low(
    tmp_path,
) -> None:
    ledger = _ledger(tmp_path)
    _write_usage_rows(ledger, api_name="daily_basic", count=190)
    manager = TushareQuotaManager(
        profile=DEFAULT_TUSHARE_2000_POINT_PROFILE,
        ledger=ledger,
    )

    decision = manager.evaluate(
        _cost("fina_indicator", 49),
        manager.current_state(),
        execution_mode="autonomous",
    )

    assert decision.status == "APPROVED_WITH_RATE_PACING"
    assert decision.safe_to_execute_now is True
    assert decision.recommended_batches[0]["request_count"] == 10


def test_quota_manager_blocks_daily_quota_per_api_without_cross_api_bleed(
    tmp_path,
) -> None:
    ledger = _ledger(tmp_path)
    _write_usage_rows(ledger, api_name="daily_basic", count=99990)
    manager = TushareQuotaManager(
        profile=DEFAULT_TUSHARE_2000_POINT_PROFILE,
        ledger=ledger,
    )
    state = manager.current_state()

    blocked = manager.evaluate(
        _cost("daily_basic", 116),
        state,
        execution_mode="autonomous",
    )
    allowed_other_api = manager.evaluate(
        _cost("fina_indicator", 49),
        state,
        execution_mode="autonomous",
    )

    assert blocked.status == "BLOCKED_BY_DAILY_QUOTA"
    assert allowed_other_api.status in {
        "APPROVED_BY_ACCOUNT_QUOTA",
        "APPROVED_WITH_RATE_PACING",
    }


def test_unknown_quota_requires_approval() -> None:
    manager = TushareQuotaManager(
        profile=TushareAccountQuotaProfile(source="unknown", confidence="LOW")
    )

    decision = manager.evaluate(
        _cost("daily_basic", 1),
        manager.current_state(),
        execution_mode="autonomous",
    )

    assert decision.status == "UNKNOWN_QUOTA_REQUIRES_APPROVAL"
    assert decision.safe_to_execute_now is False


def test_usage_ledger_records_success_failure_rate_limit_and_ignores_dry_run(
    tmp_path,
) -> None:
    ledger = _ledger(tmp_path)
    params = {"trade_date": "20240102"}
    fields = ["ts_code", "trade_date"]

    ledger.append(
        new_usage_record(
            api_name="daily_basic",
            params=params,
            fields=fields,
            status="SUCCESS",
            execution_mode="manual",
            row_count=1,
            finished_at=datetime.now(tz=UTC),
        )
    )
    ledger.append(
        new_usage_record(
            api_name="daily_basic",
            params={"trade_date": "20240103"},
            fields=fields,
            status="FAILED",
            execution_mode="manual",
            error_type="RuntimeError",
            error_message="boom",
            finished_at=datetime.now(tz=UTC),
        )
    )
    ledger.append(
        new_usage_record(
            api_name="daily_basic",
            params={"trade_date": "20240104"},
            fields=fields,
            status="RATE_LIMITED",
            execution_mode="manual",
            error_type="RateLimit",
            error_message="rate limit",
            finished_at=datetime.now(tz=UTC),
        )
    )
    ledger.append(
        new_usage_record(
            api_name="daily_basic",
            params={"trade_date": "20240105"},
            fields=fields,
            status="PLANNED",
            execution_mode="dry_run",
        )
    )

    request_hash = normalized_request_hash(
        api_name="daily_basic",
        params=params,
        fields=list(reversed(fields)),
    )
    assert ledger.usage_last_minute() == 3
    assert ledger.usage_today("daily_basic") == 3
    assert len(ledger.recent_rate_limit_errors()) == 1
    assert ledger.request_seen("daily_basic", request_hash) is True


def test_usage_ledger_uses_duckdb_and_request_id_is_idempotent(tmp_path) -> None:
    lake = DataLake(root=tmp_path / "lake", duckdb_path=tmp_path / "db.duckdb")
    ledger = TushareUsageLedger.from_data_lake(lake)
    record = new_usage_record(
        api_name="daily_basic",
        params={"trade_date": "20240102"},
        fields=["ts_code", "trade_date"],
        status="SUCCESS",
        execution_mode="manual",
        finished_at=datetime.now(tz=UTC),
    )

    ledger.append(record)
    ledger.append(record)

    assert not ledger.path.exists()
    with duckdb.connect(str(lake.duckdb_path), read_only=True) as connection:
        count = connection.execute("SELECT count(*) FROM tushare_usage_events_v1").fetchone()
    assert count == (1,)
    assert ledger.request_seen("daily_basic", record.params_hash) is True


def test_from_lake_root_is_formally_deprecated(tmp_path) -> None:
    with pytest.warns(DeprecationWarning, match="from_data_lake"):
        TushareUsageLedger.from_lake_root(tmp_path / "lake")


def test_empty_legacy_ledger_is_archived_without_usage(tmp_path) -> None:
    lake = DataLake(root=tmp_path / "lake", duckdb_path=tmp_path / "db.duckdb")
    ledger = TushareUsageLedger.from_data_lake(lake)
    ledger.path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(columns=_usage_row_columns()).to_parquet(ledger.path, index=False)

    assert ledger.usage_today_by_api() == {}
    assert not ledger.path.exists()
    assert len(list((ledger.path.parent / "archive").glob("*.parquet"))) == 1


def test_legacy_ledger_missing_required_columns_is_rejected(tmp_path) -> None:
    lake = DataLake(root=tmp_path / "lake", duckdb_path=tmp_path / "db.duckdb")
    ledger = TushareUsageLedger.from_data_lake(lake)
    ledger.path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"request_id": "incomplete"}]).to_parquet(ledger.path, index=False)

    with pytest.raises(TushareUsageLedgerCorruptError, match="missing required columns"):
        ledger.usage_today_by_api()

    assert ledger.path.exists()


def test_legacy_ledger_invalid_record_is_rejected_before_import(tmp_path) -> None:
    lake = DataLake(root=tmp_path / "lake", duckdb_path=tmp_path / "db.duckdb")
    ledger = TushareUsageLedger.from_data_lake(lake)
    record = new_usage_record(
        api_name="daily_basic",
        params={"trade_date": "20240102"},
        fields=["ts_code"],
        status="SUCCESS",
        execution_mode="manual",
    )
    payload = _usage_row(record)
    payload["status"] = "NOT_A_REAL_STATUS"
    ledger.path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([payload]).to_parquet(ledger.path, index=False)

    with pytest.raises(TushareUsageLedgerCorruptError, match="validation"):
        ledger.usage_today_by_api()

    assert ledger.path.exists()


def test_legacy_migration_reapplies_token_redaction(tmp_path) -> None:
    lake = DataLake(root=tmp_path / "lake", duckdb_path=tmp_path / "db.duckdb")
    ledger = TushareUsageLedger.from_data_lake(lake)
    record = new_usage_record(
        api_name="daily_basic",
        params={"trade_date": "20240102"},
        fields=["ts_code"],
        status="SUCCESS",
        execution_mode="manual",
        token_hash=quota_module.token_hash("top-secret"),
    )
    payload = _usage_row(record)
    payload["params_redacted"] = '{"token": "top-secret", "trade_date": "20240102"}'
    ledger.path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([payload]).to_parquet(ledger.path, index=False)

    ledger.usage_today_by_api()

    with duckdb.connect(str(lake.duckdb_path), read_only=True) as connection:
        params = connection.execute(
            "SELECT params_redacted FROM tushare_usage_events_v1"
        ).fetchone()
    assert params is not None
    assert "top-secret" not in str(params)
    assert "<redacted>" in str(params)


def test_footer_readable_but_data_page_corrupt_ledger_is_rejected(tmp_path) -> None:
    lake = DataLake(root=tmp_path / "lake", duckdb_path=tmp_path / "db.duckdb")
    ledger = TushareUsageLedger.from_data_lake(lake)
    record = new_usage_record(
        api_name="daily_basic",
        params={"trade_date": "20240102"},
        fields=["ts_code"],
        status="SUCCESS",
        execution_mode="manual",
        finished_at=datetime.now(tz=UTC),
    )
    ledger.path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([_usage_row(record)]).to_parquet(ledger.path, index=False)
    parquet = pq.ParquetFile(ledger.path)
    page_offset = parquet.metadata.row_group(0).column(0).data_page_offset
    with ledger.path.open("r+b") as handle:
        handle.seek(page_offset)
        handle.write(b"\xff" * 12)

    pq.read_metadata(ledger.path)
    with pytest.raises(Exception, match="Local Tushare usage ledger is unreadable"):
        ledger.usage_last_minute()


def test_usage_ledger_never_persists_plaintext_token(tmp_path) -> None:
    lake = DataLake(root=tmp_path / "lake", duckdb_path=tmp_path / "db.duckdb")
    ledger = TushareUsageLedger.from_data_lake(lake)
    ledger.append(
        new_usage_record(
            api_name="daily_basic",
            params={"token": "top-secret", "trade_date": "20240102"},
            fields=["ts_code"],
            status="SUCCESS",
            execution_mode="manual",
            token_hash=quota_module.token_hash("top-secret"),
            finished_at=datetime.now(tz=UTC),
        )
    )

    with duckdb.connect(str(lake.duckdb_path), read_only=True) as connection:
        stored = connection.execute(
            "SELECT params_redacted, token_hash FROM tushare_usage_events_v1"
        ).fetchone()
    assert stored is not None
    assert "top-secret" not in str(stored)
    assert "<redacted>" in str(stored)


def test_usage_ledger_serializes_independent_process_writers(tmp_path) -> None:
    lake_root = tmp_path / "lake"
    duckdb_path = tmp_path / "db.duckdb"
    context = get_context("spawn")

    with context.Pool(processes=4) as pool:
        pool.starmap(
            _append_usage_in_process,
            [(str(lake_root), str(duckdb_path), index) for index in range(8)],
        )

    with duckdb.connect(str(duckdb_path), read_only=True) as connection:
        count = connection.execute("SELECT count(*) FROM tushare_usage_events_v1").fetchone()
    assert count == (8,)


def test_usage_ledger_concurrent_appends_keep_every_record(tmp_path) -> None:
    lake = DataLake(root=tmp_path / "lake", duckdb_path=tmp_path / "db.duckdb")
    ledger = TushareUsageLedger.from_data_lake(lake)
    records = [
        new_usage_record(
            api_name="daily_basic",
            params={"trade_date": f"202401{index + 1:02d}"},
            fields=["ts_code", "trade_date"],
            status="SUCCESS",
            execution_mode="manual",
            finished_at=datetime.now(tz=UTC),
        )
        for index in range(20)
    ]

    with ThreadPoolExecutor(max_workers=8) as executor:
        list(executor.map(ledger.append, records))

    with duckdb.connect(str(lake.duckdb_path), read_only=True) as connection:
        count = connection.execute("SELECT count(*) FROM tushare_usage_events_v1").fetchone()
    assert count == (20,)


def test_healthy_legacy_ledger_is_migrated_once_and_archived(tmp_path) -> None:
    lake = DataLake(root=tmp_path / "lake", duckdb_path=tmp_path / "db.duckdb")
    ledger = TushareUsageLedger.from_data_lake(lake)
    record = new_usage_record(
        api_name="daily_basic",
        params={"trade_date": "20240102"},
        fields=["ts_code"],
        status="SUCCESS",
        execution_mode="manual",
        finished_at=datetime.now(tz=UTC),
    )
    ledger.path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([_usage_row(record)]).to_parquet(ledger.path, index=False)

    assert ledger.usage_today("daily_basic") == 1
    assert not ledger.path.exists()
    archives = list((ledger.path.parent / "archive").glob("*.parquet"))
    assert len(archives) == 1

    assert ledger.usage_today("daily_basic") == 1
    with duckdb.connect(str(lake.duckdb_path), read_only=True) as connection:
        count = connection.execute("SELECT count(*) FROM tushare_usage_events_v1").fetchone()
    assert count == (1,)
    with duckdb.connect(str(lake.duckdb_path), read_only=True) as connection:
        migration = connection.execute(
            """
            SELECT imported_rows, skipped_rows, archive_path
            FROM tushare_usage_migrations_v1
            """
        ).fetchone()
    assert migration is not None
    assert migration[0:2] == (1, 0)
    assert str(migration[2]).endswith(".parquet")


def test_corrupt_legacy_ledger_blocks_until_explicit_quarantine(tmp_path) -> None:
    lake = DataLake(root=tmp_path / "lake", duckdb_path=tmp_path / "db.duckdb")
    ledger = TushareUsageLedger.from_data_lake(lake)
    ledger.path.parent.mkdir(parents=True, exist_ok=True)
    ledger.path.write_bytes(b"PAR1broken-ledger-pagePAR1")

    with pytest.raises(TushareUsageLedgerCorruptError) as caught:
        ledger.usage_last_minute()

    assert caught.value.path == ledger.path
    assert caught.value.suggested_repair == (
        "qmt-agent data repair-tushare-ledger --quarantine-corrupt"
    )
    inspected = repair_tushare_usage_ledger(ledger, quarantine_corrupt=False)
    assert inspected["status"] == "CORRUPT"
    assert inspected["modified"] is False
    assert ledger.path.exists()

    repaired = repair_tushare_usage_ledger(ledger, quarantine_corrupt=True)
    assert repaired["status"] == "QUARANTINED"
    assert repaired["history_reset"] is True
    assert not ledger.path.exists()
    quarantine_path = tmp_path / repaired["quarantine_path"]
    sidecar_path = tmp_path / repaired["sidecar_path"]
    assert quarantine_path.exists()
    sidecar = pd.read_json(sidecar_path, typ="series")
    assert sidecar["sha256"]
    assert sidecar["original_path"].endswith("tushare_usage_ledger.parquet")
    assert "history" in repaired["warning"].lower()
    assert ledger.history_warnings() == ["TUSHARE_USAGE_HISTORY_RESET"]


def test_quarantine_keeps_source_when_history_reset_persistence_fails(
    monkeypatch,
    tmp_path,
) -> None:
    lake = DataLake(root=tmp_path / "lake", duckdb_path=tmp_path / "db.duckdb")
    ledger = TushareUsageLedger.from_data_lake(lake)
    ledger.path.parent.mkdir(parents=True, exist_ok=True)
    ledger.path.write_bytes(b"PAR1broken-ledger-pagePAR1")

    def fail_history_reset(*_args, **_kwargs):
        raise RuntimeError("simulated history reset failure")

    monkeypatch.setattr(
        ledger_migration,
        "_record_history_reset_locked",
        fail_history_reset,
        raising=False,
    )

    with pytest.raises(RuntimeError, match="history reset failure"):
        repair_tushare_usage_ledger(ledger, quarantine_corrupt=True)

    assert ledger.path.exists()
    assert list((ledger.path.parent / "corrupt").glob("*.parquet")) == []


def test_migration_audit_survives_finalization_failure(monkeypatch, tmp_path) -> None:
    lake = DataLake(root=tmp_path / "lake", duckdb_path=tmp_path / "db.duckdb")
    ledger = TushareUsageLedger.from_data_lake(lake)
    record = new_usage_record(
        api_name="daily_basic",
        params={"trade_date": "20240102"},
        fields=["ts_code"],
        status="SUCCESS",
        execution_mode="manual",
    )
    ledger.path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([_usage_row(record)]).to_parquet(ledger.path, index=False)

    def fail_finalize(*_args, **_kwargs):
        raise RuntimeError("simulated migration finalization failure")

    monkeypatch.setattr(
        ledger_migration,
        "_finalize_migration",
        fail_finalize,
        raising=False,
    )

    with pytest.raises(RuntimeError, match="migration finalization failure"):
        ledger.usage_today_by_api()

    with duckdb.connect(str(lake.duckdb_path), read_only=True) as connection:
        audit = connection.execute(
            """
            SELECT status, imported_rows, skipped_rows, archive_path
            FROM tushare_usage_migrations_v1
            """
        ).fetchone()
    assert audit is not None
    assert audit[0:3] == ("IMPORT_COMMITTED_PENDING_ARCHIVE", 1, 0)
    assert Path(str(audit[3])).exists()


def test_migration_retry_resumes_pending_archive_without_duplicate_audit(
    monkeypatch,
    tmp_path,
) -> None:
    lake = DataLake(root=tmp_path / "lake", duckdb_path=tmp_path / "db.duckdb")
    ledger = TushareUsageLedger.from_data_lake(lake)
    record = new_usage_record(
        api_name="daily_basic",
        params={"trade_date": "20240102"},
        fields=["ts_code"],
        status="SUCCESS",
        execution_mode="manual",
    )
    ledger.path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([_usage_row(record)]).to_parquet(ledger.path, index=False)
    original_replace = ledger_migration.os.replace
    failed = False

    def fail_first_archive(source, destination):
        nonlocal failed
        if Path(source) == ledger.path and not failed:
            failed = True
            raise OSError("simulated archive replace failure")
        return original_replace(source, destination)

    monkeypatch.setattr(ledger_migration.os, "replace", fail_first_archive)

    with pytest.raises(OSError, match="archive replace failure"):
        ledger.usage_today_by_api()
    assert ledger.path.exists()

    assert ledger.usage_today("daily_basic") == 0

    with duckdb.connect(str(lake.duckdb_path), read_only=True) as connection:
        migrations = connection.execute(
            """
            SELECT status, imported_rows, skipped_rows, archive_path
            FROM tushare_usage_migrations_v1
            """
        ).fetchall()
        usage_count = connection.execute(
            "SELECT count(*) FROM tushare_usage_events_v1"
        ).fetchone()
    assert len(migrations) == 1
    assert migrations[0][0:3] == ("MIGRATED", 1, 0)
    assert Path(str(migrations[0][3])).exists()
    assert usage_count == (1,)


def _usage_row(record) -> dict[str, object]:
    payload = record.model_dump(mode="json")
    payload["params_redacted"] = "{}"
    payload["fields"] = '["ts_code"]'
    return payload


def _ledger(tmp_path: Path) -> TushareUsageLedger:
    return TushareUsageLedger.from_data_lake(
        DataLake(root=tmp_path / "lake", duckdb_path=tmp_path / "qmt_agent_trader.duckdb")
    )


def _usage_row_columns() -> list[str]:
    return [
        "request_id",
        "run_id",
        "api_name",
        "params_hash",
        "params_redacted",
        "fields",
        "planned_at",
        "started_at",
        "finished_at",
        "status",
        "row_count",
        "error_type",
        "error_message",
        "token_hash",
        "execution_mode",
    ]


def _append_usage_in_process(lake_root: str, duckdb_path: str, index: int) -> None:
    lake = DataLake(root=Path(lake_root), duckdb_path=Path(duckdb_path))
    ledger = TushareUsageLedger.from_data_lake(lake)
    ledger.append(
        new_usage_record(
            api_name="daily_basic",
            params={"request": index},
            fields=["ts_code"],
            status="SUCCESS",
            execution_mode="manual",
            finished_at=datetime.now(tz=UTC),
        )
    )


def test_equivalent_request_hash_is_stable() -> None:
    left = normalized_request_hash(
        api_name="daily_basic",
        params={"b": 2, "a": 1},
        fields=["trade_date", "ts_code"],
    )
    right = normalized_request_hash(
        api_name="daily_basic",
        params={"a": 1, "b": 2},
        fields=["ts_code", "trade_date"],
    )

    assert left == right


def _cost(api_name: str, count: int) -> TushareFetchCostEstimate:
    manager = TushareQuotaManager(profile=DEFAULT_TUSHARE_2000_POINT_PROFILE)
    return manager.estimate_cost(
        [
            {
                "api_name": api_name,
                "batches": [
                    {
                        "api_name": api_name,
                        "params": {"request": index},
                        "fields": ["ts_code"],
                    }
                    for index in range(count)
                ],
            }
        ]
    )


def _write_usage_rows(
    ledger: TushareUsageLedger,
    *,
    api_name: str,
    count: int,
) -> None:
    ledger.path.parent.mkdir(parents=True, exist_ok=True)
    now = datetime.now(tz=UTC).replace(tzinfo=None)
    old = now - timedelta(hours=2)
    pd.DataFrame(
        [
            {
                "request_id": f"{api_name}-{index}",
                "run_id": "quota-test",
                "api_name": api_name,
                "params_hash": f"hash-{index}",
                "params_redacted": "{}",
                "fields": '["ts_code"]',
                "planned_at": old,
                "started_at": now,
                "finished_at": now,
                "status": "SUCCESS",
                "row_count": 1,
                "error_type": None,
                "error_message": None,
                "token_hash": None,
                "execution_mode": "manual",
            }
            for index in range(count)
        ]
    ).to_parquet(ledger.path, index=False)
