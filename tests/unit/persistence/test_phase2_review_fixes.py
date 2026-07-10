from __future__ import annotations

import gc
import multiprocessing
import time
import weakref
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from threading import Event, Thread
from typing import Any

import duckdb
import pytest

from qmt_agent_trader.data.providers.tushare.quota import (
    TushareUsageLedger,
    new_usage_record,
)
from qmt_agent_trader.data.storage import DataLake
from qmt_agent_trader.persistence import database as database_module
from qmt_agent_trader.persistence.database import DatabaseCoordinator
from qmt_agent_trader.persistence.errors import (
    StorageConflictError,
    StorageLockTimeoutError,
)
from qmt_agent_trader.persistence.initialization import initialize_persistence
from qmt_agent_trader.persistence.locks import LockManager


def _coordinated_lake(database: Path, lake_root: Path, locks: Path) -> DataLake:
    manager = LockManager(locks, timeout_seconds=3)
    return DataLake(
        lake_root,
        database,
        database_coordinator=DatabaseCoordinator(database, manager),
        lock_manager=manager,
    )


def _hold_read(
    database: str,
    locks: str,
    sql: str,
    ready: Any,
    release: Any,
) -> None:
    coordinator = DatabaseCoordinator(Path(database), LockManager(Path(locks), timeout_seconds=3))
    with coordinator.read_connection("long_read") as connection:
        connection.execute(sql).fetchone()
        ready.set()
        if not release.wait(timeout=5):
            raise TimeoutError("reader release was not signaled")


def _write_metadata(database: str, lake_root: str, locks: str) -> None:
    lake = _coordinated_lake(Path(database), Path(lake_root), Path(locks))
    initialize_persistence(lake)
    lake.record_fetch_result(
        source="tushare",
        dataset="tushare.daily",
        start_date="20240101",
        end_date="20240131",
        status="SUCCESS",
        row_count=1,
        checksum="write",
        error=None,
    )


def _append_usage(database: str, lake_root: str, locks: str) -> None:
    lake = _coordinated_lake(Path(database), Path(lake_root), Path(locks))
    initialize_persistence(lake)
    TushareUsageLedger.from_data_lake(lake).append(
        new_usage_record(
            api_name="daily",
            params={"trade_date": "20240102"},
            fields=["ts_code"],
            status="SUCCESS",
            execution_mode="manual",
            finished_at=datetime.now(tz=UTC),
        )
    )


@pytest.mark.parametrize(
    ("read_sql", "writer"),
    [
        ("SELECT count(*) FROM data_fetch_state", _write_metadata),
        ("SELECT count(*) FROM tushare_usage_events_v1", _append_usage),
    ],
)
def test_long_cross_process_read_and_coordinated_write_both_complete(
    tmp_path: Path, read_sql: str, writer: Any
) -> None:
    database, lake_root, locks = (
        tmp_path / "control.duckdb",
        tmp_path / "lake",
        tmp_path / "locks",
    )
    initialize_persistence(_coordinated_lake(database, lake_root, locks))
    context = multiprocessing.get_context("spawn")
    ready, release = context.Event(), context.Event()
    reader = context.Process(
        target=_hold_read,
        args=(str(database), str(locks), read_sql, ready, release),
    )
    reader.start()
    assert ready.wait(timeout=5)
    writing = context.Process(
        target=writer,
        args=(str(database), str(lake_root), str(locks)),
    )
    writing.start()
    time.sleep(0.2)
    release.set()
    reader.join(timeout=8)
    writing.join(timeout=8)
    assert reader.exitcode == 0
    assert writing.exitcode == 0


def test_two_cross_process_readers_overlap_without_write_lock(tmp_path: Path) -> None:
    database, lake_root, locks = (
        tmp_path / "control.duckdb",
        tmp_path / "lake",
        tmp_path / "locks",
    )
    initialize_persistence(_coordinated_lake(database, lake_root, locks))
    context = multiprocessing.get_context("spawn")
    release = context.Event()
    ready = [context.Event(), context.Event()]
    readers = [
        context.Process(
            target=_hold_read,
            args=(
                str(database),
                str(locks),
                "SELECT count(*) FROM data_fetch_state",
                item,
                release,
            ),
        )
        for item in ready
    ]
    for reader in readers:
        reader.start()
    assert all(item.wait(timeout=5) for item in ready)
    release.set()
    for reader in readers:
        reader.join(timeout=8)
        assert reader.exitcode == 0


def test_cold_readiness_checks_execute_select_only(tmp_path: Path) -> None:
    initialized = _coordinated_lake(
        tmp_path / "control.duckdb", tmp_path / "lake", tmp_path / "locks"
    )
    initialize_persistence(initialized)
    cold = _coordinated_lake(tmp_path / "control.duckdb", tmp_path / "lake", tmp_path / "locks")
    ledger = TushareUsageLedger.from_data_lake(cold)
    statements: list[str] = []
    original = cold.database_coordinator.read_connection

    @contextmanager
    def tracing(operation: str = "read", *, read_only: bool = False):
        with original(operation, read_only=read_only) as connection:

            class Proxy:
                def execute(self, sql: str, parameters: Any = None) -> Any:
                    statements.append(sql.strip())
                    return (
                        connection.execute(sql)
                        if parameters is None
                        else connection.execute(sql, parameters)
                    )

            yield Proxy()

    cold.database_coordinator.read_connection = tracing  # type: ignore[method-assign]
    assert cold.fetch_state("tushare", "missing") == []
    assert ledger.usage_today_by_api() == {}
    assert statements
    assert all(statement.upper().startswith("SELECT") for statement in statements)


def test_from_lake_root_eager_compatibility_bridge_is_end_to_end(tmp_path: Path) -> None:
    lake_root = tmp_path / "lake"
    with pytest.warns(DeprecationWarning, match="from_data_lake"):
        ledger = TushareUsageLedger.from_lake_root(lake_root)
    record = new_usage_record(
        api_name="daily",
        params={"trade_date": "20240102"},
        fields=["ts_code"],
        status="SUCCESS",
        execution_mode="manual",
        finished_at=datetime.now(tz=UTC),
    )
    ledger.append(record)
    assert ledger.request_seen("daily", record.params_hash)


def test_database_process_lock_retry_timeout_is_structured(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    coordinator = DatabaseCoordinator(
        tmp_path / "control.duckdb",
        LockManager(tmp_path / "locks", timeout_seconds=0.01),
    )

    def locked(*_args: Any, **_kwargs: Any) -> Any:
        raise duckdb.IOException("Could not set lock: Conflicting lock is held")

    monkeypatch.setattr("qmt_agent_trader.persistence.database.duckdb.connect", locked)
    with pytest.raises(StorageLockTimeoutError) as caught:
        with coordinator.read_connection("bounded_read"):
            pass
    assert caught.value.operation == "bounded_read"
    assert caught.value.recoverable is True


def _thread_coordinator(tmp_path: Path, timeout: float = 2) -> DatabaseCoordinator:
    coordinator = DatabaseCoordinator(
        tmp_path / "threaded.duckdb",
        LockManager(tmp_path / "locks", timeout_seconds=timeout),
    )
    with coordinator.write_transaction("setup") as connection:
        connection.execute("CREATE TABLE IF NOT EXISTS values_table(value INTEGER)")
    return coordinator


@pytest.mark.parametrize("holder_kind", ["reader", "writer"])
def test_same_process_reader_and_writer_wait_without_duckdb_configuration_error(
    tmp_path: Path, holder_kind: str
) -> None:
    coordinator = _thread_coordinator(tmp_path)
    held, release, waiter_done = Event(), Event(), Event()
    errors: list[BaseException] = []

    def holder() -> None:
        context = (
            coordinator.read_connection("held_reader")
            if holder_kind == "reader"
            else coordinator.write_transaction("held_writer")
        )
        with context as connection:
            connection.execute("SELECT count(*) FROM values_table").fetchone()
            held.set()
            assert release.wait(timeout=3)

    def waiter() -> None:
        try:
            context = (
                coordinator.write_transaction("waiting_writer")
                if holder_kind == "reader"
                else coordinator.read_connection("waiting_reader")
            )
            with context as connection:
                connection.execute("SELECT count(*) FROM values_table").fetchone()
        except BaseException as exc:
            errors.append(exc)
        finally:
            waiter_done.set()

    holder_thread, waiter_thread = Thread(target=holder), Thread(target=waiter)
    holder_thread.start()
    assert held.wait(timeout=2)
    waiter_thread.start()
    assert not waiter_done.wait(timeout=0.1)
    release.set()
    holder_thread.join(timeout=3)
    waiter_thread.join(timeout=3)
    assert waiter_done.is_set()
    assert errors == []


def test_same_process_multiple_readers_overlap(tmp_path: Path) -> None:
    coordinator = _thread_coordinator(tmp_path)
    ready = [Event(), Event()]
    release = Event()

    def reader(signal: Event) -> None:
        with coordinator.read_connection("parallel_reader") as connection:
            connection.execute("SELECT count(*) FROM values_table").fetchone()
            signal.set()
            assert release.wait(timeout=3)

    threads = [Thread(target=reader, args=(signal,)) for signal in ready]
    for thread in threads:
        thread.start()
    assert all(signal.wait(timeout=2) for signal in ready)
    release.set()
    for thread in threads:
        thread.join(timeout=3)
        assert not thread.is_alive()


def test_same_process_gate_timeout_and_nested_write_are_structured(tmp_path: Path) -> None:
    coordinator = _thread_coordinator(tmp_path, timeout=0.05)
    held, release = Event(), Event()

    def writer() -> None:
        with coordinator.write_transaction("held_writer"):
            held.set()
            assert release.wait(timeout=3)

    thread = Thread(target=writer)
    thread.start()
    assert held.wait(timeout=2)
    with pytest.raises(StorageLockTimeoutError):
        with coordinator.read_connection("timed_reader"):
            pass
    release.set()
    thread.join(timeout=3)

    with coordinator.write_transaction("outer_writer"):
        with pytest.raises(StorageConflictError):
            with coordinator.read_connection("nested_reader"):
                pass


def test_timed_out_preferred_writer_wakes_blocked_reader_immediately(
    tmp_path: Path,
) -> None:
    database = tmp_path / "abandoned-writer.duckdb"
    setup = DatabaseCoordinator(database, LockManager(tmp_path / "locks", timeout_seconds=1))
    with setup.write_transaction("setup") as connection:
        connection.execute("CREATE TABLE values_table(value INTEGER)")

    reader_a = DatabaseCoordinator(database, LockManager(tmp_path / "locks", timeout_seconds=1))
    writer = DatabaseCoordinator(database, LockManager(tmp_path / "locks", timeout_seconds=0.12))
    reader_b = DatabaseCoordinator(database, LockManager(tmp_path / "locks", timeout_seconds=0.8))
    a_held, release_a = Event(), Event()
    writer_started, writer_timed_out, b_acquired = Event(), Event(), Event()

    def hold_reader_a() -> None:
        with reader_a.read_connection("reader_a"):
            a_held.set()
            assert release_a.wait(timeout=2)

    def wait_writer() -> None:
        writer_started.set()
        with pytest.raises(StorageLockTimeoutError):
            with writer.write_transaction("abandoned_writer"):
                pass
        writer_timed_out.set()

    def wait_reader_b() -> None:
        with reader_b.read_connection("reader_b"):
            b_acquired.set()

    a_thread = Thread(target=hold_reader_a)
    writer_thread = Thread(target=wait_writer)
    b_thread = Thread(target=wait_reader_b)
    a_thread.start()
    assert a_held.wait(timeout=1)
    writer_thread.start()
    assert writer_started.wait(timeout=1)
    time.sleep(0.03)
    b_thread.start()
    assert writer_timed_out.wait(timeout=1)
    assert b_acquired.wait(timeout=0.2)
    release_a.set()
    for thread in (a_thread, writer_thread, b_thread):
        thread.join(timeout=2)
        assert not thread.is_alive()


def test_unused_database_gate_is_released_from_global_registry(tmp_path: Path) -> None:
    database = (tmp_path / "lifecycle.duckdb").resolve()
    coordinator = DatabaseCoordinator(database, LockManager(tmp_path / "locks", timeout_seconds=1))
    gate_reference = weakref.ref(coordinator._gate)
    assert database in database_module._database_gates

    del coordinator
    gc.collect()

    assert gate_reference() is None
    assert database not in database_module._database_gates
