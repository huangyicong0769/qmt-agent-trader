from __future__ import annotations

import multiprocessing
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import duckdb
import pytest

from qmt_agent_trader.data.providers.tushare.quota import (
    TushareUsageLedger,
    new_usage_record,
)
from qmt_agent_trader.data.storage import DataLake, _stable_hash
from qmt_agent_trader.persistence.database import DatabaseCoordinator
from qmt_agent_trader.persistence.initialization import initialize_persistence
from qmt_agent_trader.persistence.locks import LockManager


def _initialize_from_process(database: str, lake_root: str, locks: str) -> None:
    manager = LockManager(Path(locks), timeout_seconds=10)
    coordinator = DatabaseCoordinator(Path(database), manager)
    lake = DataLake(
        Path(lake_root),
        Path(database),
        database_coordinator=coordinator,
        lock_manager=manager,
    )
    initialize_persistence(lake)


def _write_from_process(database: str, lake_root: str, locks: str, index: int) -> None:
    manager = LockManager(Path(locks), timeout_seconds=10)
    coordinator = DatabaseCoordinator(Path(database), manager)
    lake = DataLake(
        Path(lake_root),
        Path(database),
        database_coordinator=coordinator,
        lock_manager=manager,
    )
    initialize_persistence(lake)
    lake.record_fetch_metadata(
        source="tushare",
        dataset_id="tushare.daily",
        api_name="daily",
        endpoint_id="daily",
        params={"trade_date": "20240102"},
        fields=["ts_code"],
        symbols=[],
        coverage_start="20240102",
        coverage_end="20240102",
        row_count=index,
        checksum=str(index),
        status=f"success-{index}",
        error=None,
    )
    lake.record_fetch_result(
        source="tushare",
        dataset="tushare.daily",
        start_date="20240101",
        end_date="20240131",
        status=f"success-{index}",
        row_count=index,
        checksum=str(index),
        error=None,
    )
    TushareUsageLedger.from_data_lake(lake).append(
        new_usage_record(
            api_name="daily",
            params={"process": index},
            fields=["ts_code"],
            status="SUCCESS",
            execution_mode="manual",
            finished_at=datetime.now(tz=UTC),
        )
    )


def _lake(tmp_path: Path) -> DataLake:
    manager = LockManager(tmp_path / "locks", timeout_seconds=10)
    coordinator = DatabaseCoordinator(tmp_path / "control.duckdb", manager)
    return DataLake(
        tmp_path / "lake",
        tmp_path / "control.duckdb",
        database_coordinator=coordinator,
        lock_manager=manager,
    )


def test_startup_migrations_are_idempotent_across_processes(tmp_path: Path) -> None:
    database = tmp_path / "control.duckdb"
    lake_root = tmp_path / "lake"
    locks = tmp_path / "locks"
    context = multiprocessing.get_context("spawn")
    processes = [
        context.Process(
            target=_initialize_from_process,
            args=(str(database), str(lake_root), str(locks)),
        )
        for _ in range(4)
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=20)
        assert process.exitcode == 0

    with duckdb.connect(str(database), read_only=True) as connection:
        rows = connection.execute(
            "SELECT migration_id, count(*) FROM storage_schema_migrations "
            "WHERE status='APPLIED' GROUP BY migration_id"
        ).fetchall()
    assert {migration_id for migration_id, _ in rows} == {
        "data-fetch-metadata-v1",
        "data-fetch-state-primary-keys-v2",
        "tushare-usage-store-v1",
    }
    assert all(count == 1 for _, count in rows)


def test_upgrade_from_legacy_no_primary_key_state_preserves_latest_rows(
    tmp_path: Path,
) -> None:
    lake = _lake(tmp_path)
    params = {"trade_date": "20240102"}
    fields = ["ts_code"]
    symbols: list[str] = []
    hashes = [_stable_hash(params), _stable_hash(fields), _stable_hash(symbols)]
    with lake.database_coordinator.write_transaction("create_legacy_schema") as connection:
        connection.execute(
            """
            CREATE TABLE data_fetch_state_v2 (
                source TEXT, dataset_id TEXT, api_name TEXT, endpoint_id TEXT,
                param_hash TEXT, fields_hash TEXT, symbols_hash TEXT,
                fetched_at TIMESTAMP, coverage_start TEXT, coverage_end TEXT,
                row_count BIGINT, checksum TEXT, status TEXT, error TEXT
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE data_fetch_events_v2 (
                source TEXT, dataset_id TEXT, api_name TEXT, endpoint_id TEXT,
                param_hash TEXT, fields_hash TEXT, symbols_hash TEXT,
                fetched_at TIMESTAMP, coverage_start TEXT, coverage_end TEXT,
                row_count BIGINT, checksum TEXT, status TEXT, error TEXT
            )
            """
        )
        connection.execute(
            """
            INSERT INTO data_fetch_events_v2 VALUES
            ('tushare','tushare.daily','daily','daily',?,?,?,
             '2024-01-01','20240101','20240101',1,'event-v2','OLD',NULL)
            """,
            hashes,
        )
        connection.execute(
            """
            INSERT INTO data_fetch_state_v2 VALUES
            ('tushare','tushare.daily','daily','daily',?,?,?,
             '2024-01-01',NULL,NULL,1,'old','OLD',NULL),
            ('tushare','tushare.daily','daily','daily',?,?,?,
             '2024-01-02',NULL,NULL,2,'new','NEW',NULL)
            """,
            [*hashes, *hashes],
        )
        connection.execute(
            """
            CREATE TABLE data_fetch_events (
                source TEXT, dataset TEXT, start_date TEXT, end_date TEXT,
                status TEXT, row_count BIGINT, checksum TEXT,
                updated_at TIMESTAMP, error TEXT
            )
            """
        )
        connection.execute(
            """
            INSERT INTO data_fetch_events VALUES
            ('tushare','tushare.daily','20240101','20240131',
             'OLD',1,'event-v1','2024-01-01',NULL)
            """
        )
        connection.execute(
            """
            CREATE TABLE data_fetch_state (
                source TEXT, dataset TEXT, start_date TEXT, end_date TEXT,
                status TEXT, row_count BIGINT, checksum TEXT,
                updated_at TIMESTAMP, error TEXT
            )
            """
        )
        connection.execute(
            """
            INSERT INTO data_fetch_state VALUES
            ('tushare','tushare.daily','20240101','20240131','OLD',1,'old','2024-01-01',NULL),
            ('tushare','tushare.daily','20240101','20240131','NEW',2,'new','2024-01-02',NULL)
            """
        )

    initialize_persistence(lake)
    lake.record_fetch_result(
        source="tushare",
        dataset="tushare.daily",
        start_date="20240101",
        end_date="20240131",
        status="LATEST",
        row_count=3,
        checksum="latest",
        error=None,
    )
    lake.record_fetch_metadata(
        source="tushare",
        dataset_id="tushare.daily",
        api_name="daily",
        endpoint_id="daily",
        params=params,
        fields=fields,
        symbols=symbols,
        coverage_start="20240102",
        coverage_end="20240102",
        row_count=3,
        checksum="latest-v2",
        status="LATEST",
        error=None,
    )
    with lake.database_coordinator.read_connection("verify_upgrade") as connection:
        assert connection.execute(
            "SELECT count(*), max(checksum) FROM data_fetch_state_v2"
        ).fetchone() == (1, "latest-v2")
        assert connection.execute(
            "SELECT count(*), max(checksum) FROM data_fetch_state"
        ).fetchone() == (1, "latest")
        assert connection.execute(
            "SELECT checksum FROM data_fetch_events_v2 ORDER BY fetched_at"
        ).fetchall() == [("event-v2",), ("latest-v2",)]
        assert connection.execute(
            "SELECT checksum FROM data_fetch_events ORDER BY updated_at"
        ).fetchall() == [("event-v1",), ("latest",)]
        primary_keys = connection.execute(
            "SELECT table_name FROM duckdb_constraints() "
            "WHERE constraint_type='PRIMARY KEY' "
            "AND table_name IN ('data_fetch_state_v2', 'data_fetch_state')"
        ).fetchall()
        assert {row[0] for row in primary_keys} == {
            "data_fetch_state_v2",
            "data_fetch_state",
        }


def test_no_primary_key_dedupe_tie_break_is_insertion_order_independent(
    tmp_path: Path,
) -> None:
    selected: list[tuple[Any, ...]] = []
    variants = [
        ("20240101", "20240131", 7, "same", "SAME", "alpha"),
        ("20240102", "20240201", 7, "same", "SAME", "omega"),
    ]
    for index, ordered in enumerate((variants, list(reversed(variants)))):
        root = tmp_path / str(index)
        lake = _lake(root)
        with lake.database_coordinator.write_transaction("old_schema") as connection:
            connection.execute(
                """
                CREATE TABLE data_fetch_state_v2 (
                    source TEXT, dataset_id TEXT, api_name TEXT, endpoint_id TEXT,
                    param_hash TEXT, fields_hash TEXT, symbols_hash TEXT,
                    fetched_at TIMESTAMP, coverage_start TEXT, coverage_end TEXT,
                    row_count BIGINT, checksum TEXT, status TEXT, error TEXT
                )
                """
            )
            connection.executemany(
                "INSERT INTO data_fetch_state_v2 VALUES "
                "('tushare','tushare.daily','daily','daily','p','f','s',"
                "'2024-01-02',?,?,?,?,?,?)",
                ordered,
            )
            connection.execute(
                """
                CREATE TABLE data_fetch_state (
                    source TEXT, dataset TEXT, start_date TEXT, end_date TEXT,
                    status TEXT, row_count BIGINT, checksum TEXT,
                    updated_at TIMESTAMP, error TEXT
                )
                """
            )
        initialize_persistence(lake)
        with lake.database_coordinator.read_connection("selected") as connection:
            row = connection.execute(
                "SELECT coverage_start, coverage_end, row_count, checksum, status, error "
                "FROM data_fetch_state_v2"
            ).fetchone()
        assert row is not None
        selected.append(tuple(row))

    assert selected == [variants[1], variants[1]]


def test_fetch_state_and_event_rollback_together(tmp_path: Path) -> None:
    lake = _lake(tmp_path)
    initialize_persistence(lake)
    coordinator = lake.database_coordinator
    original = coordinator.write_transaction

    class FailingConnection:
        def __init__(self, connection: Any) -> None:
            self.connection = connection

        def execute(self, sql: str, parameters: Any = None) -> Any:
            if "INSERT INTO data_fetch_events_v2" in sql:
                raise RuntimeError("injected event failure")
            if parameters is None:
                return self.connection.execute(sql)
            return self.connection.execute(sql, parameters)

    @contextmanager
    def failing(operation: str = "write") -> Iterator[Any]:
        with original(operation) as connection:
            yield FailingConnection(connection)

    coordinator.write_transaction = failing  # type: ignore[method-assign]
    with pytest.raises(RuntimeError, match="injected event failure"):
        lake.record_fetch_metadata(
            source="tushare",
            dataset_id="tushare.daily",
            api_name="daily",
            endpoint_id="daily",
            params={"trade_date": "20240102"},
            fields=["ts_code"],
            symbols=[],
            coverage_start="20240102",
            coverage_end="20240102",
            row_count=1,
            checksum="abc",
            status="SUCCESS",
            error=None,
        )

    with coordinator.read_connection("verify_rollback") as connection:
        assert connection.execute("SELECT count(*) FROM data_fetch_state_v2").fetchone() == (0,)
        assert connection.execute("SELECT count(*) FROM data_fetch_events_v2").fetchone() == (0,)


def test_concurrent_metadata_and_usage_writes_share_one_database_lock(
    tmp_path: Path,
) -> None:
    database = tmp_path / "control.duckdb"
    lake_root = tmp_path / "lake"
    locks = tmp_path / "locks"
    _initialize_from_process(str(database), str(lake_root), str(locks))
    context = multiprocessing.get_context("spawn")
    processes = [
        context.Process(
            target=_write_from_process,
            args=(str(database), str(lake_root), str(locks), index),
        )
        for index in range(8)
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=30)
        assert process.exitcode == 0

    with duckdb.connect(str(database), read_only=True) as connection:
        assert connection.execute("SELECT count(*) FROM data_fetch_state").fetchone() == (1,)
        assert connection.execute("SELECT count(*) FROM data_fetch_events").fetchone() == (8,)
        assert connection.execute("SELECT count(*) FROM data_fetch_state_v2").fetchone() == (1,)
        assert connection.execute("SELECT count(*) FROM data_fetch_events_v2").fetchone() == (8,)
        assert connection.execute("SELECT count(*) FROM tushare_usage_events_v1").fetchone() == (8,)


def test_normal_reads_do_not_run_schema_or_legacy_migration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    lake = _lake(tmp_path)
    initialize_persistence(lake)
    ledger = TushareUsageLedger.from_data_lake(lake)

    def reject(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("normal read attempted initialization")

    monkeypatch.setattr(lake, "ensure_fetch_tables", reject)
    monkeypatch.setattr(ledger, "ensure_tables", reject)
    monkeypatch.setattr(
        "qmt_agent_trader.data.providers.tushare.ledger_migration.migrate_legacy_usage_ledger",
        reject,
    )

    assert lake.fetch_state("tushare", "missing") == []
    assert lake.fetch_events("tushare", "missing") == []
    assert ledger.usage_today_by_api() == {}
    assert ledger.request_seen("daily", "missing") is False
