from __future__ import annotations

import json
import multiprocessing
import os
from pathlib import Path
from threading import Thread

import duckdb
import pandas as pd
import pytest
from filelock import Timeout
from pydantic import BaseModel

from qmt_agent_trader.core.config import Settings
from qmt_agent_trader.data.providers.tushare.quota import TushareUsageLedger
from qmt_agent_trader.data.storage import DataLake
from qmt_agent_trader.persistence.atomic_files import AtomicFileStore
from qmt_agent_trader.persistence.database import DatabaseCoordinator
from qmt_agent_trader.persistence.errors import (
    StorageConflictError,
    StorageError,
    StorageLockTimeoutError,
    StorageValidationError,
)
from qmt_agent_trader.persistence.initialization import initialize_persistence
from qmt_agent_trader.persistence.locks import LockManager
from qmt_agent_trader.persistence.migrations import Migration, MigrationRegistry
from qmt_agent_trader.persistence.paths import PersistencePaths


class _Document(BaseModel):
    version: int


def _insert_from_process(database: str, locks: str, value: int) -> None:
    coordinator = DatabaseCoordinator(Path(database), LockManager(Path(locks)))
    with coordinator.write_transaction("process_insert") as connection:
        connection.execute("INSERT INTO serialized VALUES (?)", [value])


def _apply_migration_from_process(database: str, locks: str) -> None:
    coordinator = DatabaseCoordinator(Path(database), LockManager(Path(locks)))
    migration = Migration(
        "concurrent-001",
        "core",
        1,
        "insert once",
        lambda connection: connection.execute("INSERT INTO migration_effect VALUES (1)"),
        implementation="insert-v1",
    )
    MigrationRegistry(coordinator).apply([migration])


def test_paths_are_absolute_cwd_independent_and_do_not_create_directories(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = tmp_path / "project"
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)

    paths = PersistencePaths.from_settings(
        Settings(project_root=project, data_dir=Path("state"), log_dir=Path("telemetry"))
    )

    assert paths.project_root == project.resolve()
    assert paths.data_root == (project / "state").resolve()
    assert paths.lake_root == (project / "state/lake").resolve()
    assert paths.control_db_path == (project / "state/qmt_agent_trader.duckdb").resolve()
    assert paths.audit_root == (project / "telemetry/audit").resolve()
    assert paths.locks_root == (project / "state/locks").resolve()
    assert not project.exists()


def test_storage_errors_have_safe_structured_attributes(tmp_path: Path) -> None:
    secret = "token-super-secret"
    cause = RuntimeError(secret)
    error = StorageValidationError(
        store_name="artifacts",
        path=tmp_path / "record.json",
        operation="write_json",
        reason="schema validation failed",
        recoverable=True,
        suggested_repair="fix the document",
        original_error=cause,
    )

    assert error.store_name == "artifacts"
    assert error.path == (tmp_path / "record.json").resolve()
    assert error.operation == "write_json"
    assert error.reason == "schema validation failed"
    assert error.recoverable is True
    assert error.suggested_repair == "fix the document"
    assert error.original_error_type == "RuntimeError"
    assert secret not in str(error)
    assert error.__cause__ is cause


def test_lock_names_are_deterministic_safe_and_enforce_order(tmp_path: Path) -> None:
    manager = LockManager(tmp_path / "locks", timeout_seconds=0.01)
    first = manager.lock_path_for_resource(tmp_path / "a/../data.json")
    second = manager.lock_path_for_resource(tmp_path / "data.json")
    hostile = manager.lock_path_for_resource("../../outside/secret")

    assert first == second
    assert hostile.parent == (tmp_path / "locks").resolve()
    assert ".." not in hostile.name and "/" not in hostile.name
    with manager.database_write_lock(tmp_path / "database.duckdb"):
        with pytest.raises(StorageConflictError, match="lock order"):
            with manager.resource_lock("late-resource"):
                pass


def test_path_and_string_resource_aliases_share_one_lock(tmp_path: Path) -> None:
    manager = LockManager(tmp_path / "locks")
    monkey_path = tmp_path / "data.json"
    assert manager.lock_path_for_resource(monkey_path) == manager.lock_path_for_resource(
        str(monkey_path)
    )
    monkey_relative = Path("data.json")
    assert manager.lock_path_for_resource(monkey_relative) == manager.lock_path_for_resource(
        "data.json"
    )


def test_lock_timeout_is_mapped(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    manager = LockManager(tmp_path / "locks", timeout_seconds=0.01)

    def fail(*args: object, **kwargs: object) -> object:
        raise Timeout("busy")

    monkeypatch.setattr("qmt_agent_trader.persistence.locks.FileLock.acquire", fail)
    with pytest.raises(StorageLockTimeoutError) as caught:
        with manager.resource_lock("resource"):
            pass
    assert caught.value.operation == "acquire_resource_lock"


def test_atomic_file_apis_validate_cleanup_create_only_and_append_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manager = LockManager(tmp_path / "locks")
    store = AtomicFileStore(manager)
    target = tmp_path / "nested/record.json"
    store.write_json(target, {"version": 1}, validator=lambda value: value["version"] == 1)
    assert json.loads(target.read_text()) == {"version": 1}
    with pytest.raises(StorageConflictError):
        store.write_text(target, "replacement", create_only=True)
    with pytest.raises(StorageValidationError):
        store.write_json(target, {"version": 2}, validator=lambda value: False)
    assert json.loads(target.read_text()) == {"version": 1}

    def explode(stage: str, path: Path) -> None:
        if stage == "before_replace":
            raise OSError("injected")

    with pytest.raises(Exception, match="atomic write failed"):
        store.write_bytes(tmp_path / "failure.bin", b"payload", fault_hook=explode)
    assert not list(tmp_path.glob(".*.tmp"))

    calls: list[bytes] = []
    original_write = __import__("os").write

    def record_write(fd: int, data: bytes) -> int:
        calls.append(data)
        return original_write(fd, data)

    monkeypatch.setattr("qmt_agent_trader.persistence.atomic_files.os.write", record_write)
    stream = tmp_path / "audit/events.jsonl"
    store.append_jsonl(stream, {"event": "ok"})
    assert calls == [b'{"event":"ok"}\n']
    assert json.loads(stream.read_text()) == {"event": "ok"}


def test_atomic_parquet_and_locked_json_update(tmp_path: Path) -> None:
    store = AtomicFileStore(LockManager(tmp_path / "locks"))
    parquet = tmp_path / "frame.parquet"
    store.write_parquet(parquet, pd.DataFrame({"value": [1, 2]}))
    assert pd.read_parquet(parquet)["value"].tolist() == [1, 2]
    document = tmp_path / "counter.json"
    store.write_json(document, {"count": 0})
    result = store.update_json(document, lambda value: {"count": value["count"] + 1})
    assert result == {"count": 1}
    assert json.loads(document.read_text()) == result


def test_atomic_json_and_yaml_support_model_validation(tmp_path: Path) -> None:
    store = AtomicFileStore(LockManager(tmp_path / "locks"))
    json_path = tmp_path / "model.json"
    yaml_path = tmp_path / "model.yaml"
    store.write_json(json_path, {"version": 1}, model=_Document)
    store.write_yaml(yaml_path, _Document(version=2), model=_Document)
    assert json.loads(json_path.read_text()) == {"version": 1}
    assert "version: 2" in yaml_path.read_text()
    with pytest.raises(StorageValidationError):
        store.write_json(json_path, {"version": "invalid"}, model=_Document)


def test_jsonl_partial_write_restores_original_length(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = AtomicFileStore(LockManager(tmp_path / "locks"))
    stream = tmp_path / "events.jsonl"
    original = b'{"event":"existing"}\n'
    stream.write_bytes(original)
    real_write = os.write

    def short_write(descriptor: int, data: bytes) -> int:
        prefix = data[: len(data) // 2]
        real_write(descriptor, prefix)
        return len(prefix)

    monkeypatch.setattr("qmt_agent_trader.persistence.atomic_files.os.write", short_write)
    with pytest.raises(StorageError, match="append failed"):
        store.append_jsonl(stream, {"event": "partial"})
    assert stream.read_bytes() == original


def test_jsonl_rollback_failure_remains_structured_and_secret_safe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = AtomicFileStore(LockManager(tmp_path / "locks"))
    stream = tmp_path / "events.jsonl"
    stream.write_bytes(b'{"event":"existing"}\n')
    real_write = os.write

    def short_write(descriptor: int, data: bytes) -> int:
        return real_write(descriptor, data[:1])

    def rollback_failure(descriptor: int, length: int) -> None:
        raise OSError("rollback-secret-value")

    monkeypatch.setattr("qmt_agent_trader.persistence.atomic_files.os.write", short_write)
    monkeypatch.setattr(
        "qmt_agent_trader.persistence.atomic_files.os.ftruncate", rollback_failure
    )
    with pytest.raises(StorageError) as caught:
        store.append_jsonl(stream, {"event": "partial"})
    assert caught.value.recoverable is False
    assert caught.value.original_append_error_type == "OSError"
    assert caught.value.rollback_error_type == "OSError"
    assert "rollback-secret-value" not in str(caught.value)


def test_database_coordinator_commit_rollback_and_unlocked_read(tmp_path: Path) -> None:
    manager = LockManager(tmp_path / "locks")
    coordinator = DatabaseCoordinator(tmp_path / "control.duckdb", manager)
    with coordinator.write_transaction("create") as connection:
        connection.execute("CREATE TABLE values_table(value INTEGER)")
        connection.execute("INSERT INTO values_table VALUES (1)")
    with pytest.raises(RuntimeError, match="rollback"):
        with coordinator.write_transaction("failing") as connection:
            connection.execute("INSERT INTO values_table VALUES (2)")
            raise RuntimeError("rollback")
    with coordinator.read_connection("read") as connection:
        assert connection.execute("SELECT value FROM values_table").fetchall() == [(1,)]
    assert manager.active_lock_kinds == ()


def test_database_coordinator_serializes_threads_and_processes(tmp_path: Path) -> None:
    database = tmp_path / "control.duckdb"
    locks = tmp_path / "locks"
    coordinator = DatabaseCoordinator(database, LockManager(locks))
    with coordinator.write_transaction("initialize") as connection:
        connection.execute("CREATE TABLE serialized(value INTEGER)")

    def insert(value: int) -> None:
        with coordinator.write_transaction("thread_insert") as connection:
            connection.execute("INSERT INTO serialized VALUES (?)", [value])

    threads = [Thread(target=insert, args=(value,)) for value in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    context = multiprocessing.get_context("spawn")
    processes = [
        context.Process(target=_insert_from_process, args=(str(database), str(locks), value))
        for value in range(4, 6)
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=10)
        assert process.exitcode == 0
    with coordinator.read_connection("verify") as connection:
        values = connection.execute("SELECT value FROM serialized ORDER BY value").fetchall()
    assert values == [(value,) for value in range(6)]


def test_lake_and_ledger_share_injected_coordination(tmp_path: Path) -> None:
    manager = LockManager(tmp_path / "locks")
    coordinator = DatabaseCoordinator(tmp_path / "control.duckdb", manager)
    lake = DataLake(
        tmp_path / "lake",
        coordinator.database_path,
        database_coordinator=coordinator,
        lock_manager=manager,
    )
    ledger = TushareUsageLedger.from_data_lake(lake)
    assert lake.database_coordinator is coordinator
    assert ledger.database_coordinator is coordinator
    assert ledger.lock_manager is manager


def test_data_lake_and_ledger_reject_inconsistent_injected_coordination(
    tmp_path: Path,
) -> None:
    first_manager = LockManager(tmp_path / "locks-a")
    second_manager = LockManager(tmp_path / "locks-b")
    coordinator = DatabaseCoordinator(tmp_path / "first.duckdb", first_manager)
    with pytest.raises(ValueError, match="database path"):
        DataLake(
            tmp_path / "lake",
            tmp_path / "second.duckdb",
            database_coordinator=coordinator,
        )
    with pytest.raises(ValueError, match="lock manager"):
        DataLake(
            tmp_path / "lake",
            coordinator.database_path,
            database_coordinator=coordinator,
            lock_manager=second_manager,
        )
    with pytest.raises(ValueError, match="database path"):
        TushareUsageLedger(
            duckdb_path=tmp_path / "second.duckdb",
            legacy_parquet_path=tmp_path / "legacy.parquet",
            database_coordinator=coordinator,
        )
    with pytest.raises(ValueError, match="lock manager"):
        TushareUsageLedger(
            duckdb_path=coordinator.database_path,
            legacy_parquet_path=tmp_path / "legacy.parquet",
            database_coordinator=coordinator,
            lock_manager=second_manager,
        )


def test_migrations_dry_run_idempotency_and_failed_audit(tmp_path: Path) -> None:
    coordinator = DatabaseCoordinator(
        tmp_path / "control.duckdb", LockManager(tmp_path / "locks")
    )
    registry = MigrationRegistry(coordinator)
    good = Migration(
        "core-001", "core", 1, "create sample",
        lambda con: con.execute("CREATE TABLE sample(value INTEGER)"),
        implementation="create-sample-v1",
    )
    assert registry.apply([good], dry_run=True) == ["core-001"]
    assert registry.apply([good]) == ["core-001"]
    assert registry.apply([good]) == []

    changed = Migration(
        "core-001", "core", 1, "create sample", lambda con: None,
        implementation="changed-implementation",
    )
    with pytest.raises(StorageConflictError, match="checksum"):
        registry.apply([changed])

    bad = Migration(
        "core-002", "core", 2, "fail",
        lambda con: (_ for _ in ()).throw(RuntimeError("migration broke")),
        implementation="always-fail-v1",
    )
    with pytest.raises(Exception, match="migration failed"):
        registry.apply([bad])
    with coordinator.read_connection("audit") as connection:
        row = connection.execute(
            "SELECT status, error_message FROM storage_schema_migrations "
            "WHERE migration_id = 'core-002'"
        ).fetchone()
    assert row is not None and row[0] == "FAILED" and "RuntimeError" in row[1]


def test_migration_requires_non_empty_immutable_implementation() -> None:
    with pytest.raises(TypeError):
        Migration("missing-001", "core", 1, "missing", lambda connection: None)
    with pytest.raises(ValueError, match="implementation"):
        Migration(
            "empty-001", "core", 1, "empty", lambda connection: None,
            implementation="",
        )

    def closure(value: int) -> object:
        return lambda connection: connection.execute("SELECT ?", [value])

    first = Migration(
        "closure-001", "core", 1, "closure", closure(1), implementation="closure-v1"
    )
    second = Migration(
        "closure-001", "core", 1, "closure", closure(2), implementation="closure-v2"
    )
    assert first.checksum != second.checksum


def test_migration_dry_run_has_no_storage_side_effects(tmp_path: Path) -> None:
    database = tmp_path / "absent/control.duckdb"
    locks = tmp_path / "locks"
    registry = MigrationRegistry(DatabaseCoordinator(database, LockManager(locks)))
    migration = Migration(
        "dry-001", "core", 1, "dry", lambda connection: None, implementation="dry-v1"
    )
    assert registry.apply([migration], dry_run=True) == ["dry-001"]
    assert not database.exists()
    assert not locks.exists()

    existing_database = tmp_path / "existing.duckdb"
    existing_coordinator = DatabaseCoordinator(
        existing_database, LockManager(tmp_path / "existing-locks")
    )
    existing_coordinator.initialize()
    before = existing_database.stat().st_mtime_ns
    existing_registry = MigrationRegistry(existing_coordinator)
    assert existing_registry.apply([migration], dry_run=True) == ["dry-001"]
    assert existing_database.stat().st_mtime_ns == before
    with existing_coordinator.read_connection("verify_no_table") as connection:
        tables = connection.execute("SHOW TABLES").fetchall()
    assert ("storage_schema_migrations",) not in tables


def test_migration_dry_run_propagates_unrelated_catalog_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    database = tmp_path / "control.duckdb"
    coordinator = DatabaseCoordinator(database, LockManager(tmp_path / "locks"))
    coordinator.initialize()
    registry = MigrationRegistry(coordinator)
    migration = Migration(
        "dry-002", "core", 1, "dry", lambda connection: None, implementation="dry-v2"
    )
    catalog_error = duckdb.CatalogException("unrelated catalog failure")

    def fail(*args: object, **kwargs: object) -> object:
        raise StorageError(
            store_name="control_db",
            database_path=database,
            operation="dry_run_migrations",
            reason="database operation failed",
            original_error=catalog_error,
        )

    monkeypatch.setattr(coordinator, "read_connection", fail)
    with pytest.raises(StorageError):
        registry.apply([migration], dry_run=True)


def test_concurrent_migration_is_applied_once(tmp_path: Path) -> None:
    database = tmp_path / "control.duckdb"
    locks = tmp_path / "locks"
    coordinator = DatabaseCoordinator(database, LockManager(locks))
    with coordinator.write_transaction("setup") as connection:
        connection.execute("CREATE TABLE migration_effect(value INTEGER)")
    context = multiprocessing.get_context("spawn")
    processes = [
        context.Process(target=_apply_migration_from_process, args=(str(database), str(locks)))
        for _ in range(2)
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=10)
        assert process.exitcode == 0
    with coordinator.read_connection("verify") as connection:
        assert connection.execute("SELECT count(*) FROM migration_effect").fetchone() == (1,)


def test_current_schema_version_propagates_unexpected_storage_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    coordinator = DatabaseCoordinator(
        tmp_path / "control.duckdb", LockManager(tmp_path / "locks")
    )

    def fail(*args: object, **kwargs: object) -> object:
        raise StorageError(
            store_name="control_db",
            database_path=coordinator.database_path,
            operation="current_schema_version",
            reason="database corrupt",
        )

    monkeypatch.setattr(coordinator, "read_connection", fail)
    with pytest.raises(StorageError, match="database corrupt"):
        coordinator.current_schema_version()


def test_ledger_reads_do_not_run_ddl_after_locked_initialization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    lake = DataLake(tmp_path / "lake", tmp_path / "control.duckdb")
    initialize_persistence(lake)
    ledger = TushareUsageLedger.from_data_lake(lake)

    def reject_ddl(connection: object) -> None:
        raise AssertionError("read path attempted DDL")

    monkeypatch.setattr(ledger, "ensure_tables", reject_ddl)
    assert ledger.usage_today_by_api() == {}
    assert ledger.request_seen("daily", "missing") is False


def test_migrated_modules_have_no_direct_duckdb_connect() -> None:
    root = Path(__file__).parents[3] / "src/qmt_agent_trader/data"
    migrated = [
        root / "storage.py",
        root / "providers/tushare/quota.py",
        root / "providers/tushare/ledger_migration.py",
    ]
    assert all("duckdb.connect" not in path.read_text() for path in migrated)
