from __future__ import annotations

import json
import multiprocessing
from pathlib import Path
from threading import Thread

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
    StorageLockTimeoutError,
    StorageValidationError,
)
from qmt_agent_trader.persistence.locks import LockManager
from qmt_agent_trader.persistence.migrations import Migration, MigrationRegistry
from qmt_agent_trader.persistence.paths import PersistencePaths


class _Document(BaseModel):
    version: int


def _insert_from_process(database: str, locks: str, value: int) -> None:
    coordinator = DatabaseCoordinator(Path(database), LockManager(Path(locks)))
    with coordinator.write_transaction("process_insert") as connection:
        connection.execute("INSERT INTO serialized VALUES (?)", [value])


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


def test_migrations_dry_run_idempotency_and_failed_audit(tmp_path: Path) -> None:
    coordinator = DatabaseCoordinator(
        tmp_path / "control.duckdb", LockManager(tmp_path / "locks")
    )
    registry = MigrationRegistry(coordinator)
    good = Migration("core-001", "core", 1, "create sample", lambda con: con.execute(
        "CREATE TABLE sample(value INTEGER)"
    ))
    assert registry.apply([good], dry_run=True) == ["core-001"]
    assert registry.apply([good]) == ["core-001"]
    assert registry.apply([good]) == []

    changed = Migration("core-001", "core", 1, "changed", lambda con: None)
    with pytest.raises(StorageConflictError, match="checksum"):
        registry.apply([changed])

    bad = Migration("core-002", "core", 2, "fail", lambda con: (_ for _ in ()).throw(
        RuntimeError("migration broke")
    ))
    with pytest.raises(Exception, match="migration failed"):
        registry.apply([bad])
    with coordinator.read_connection("audit") as connection:
        row = connection.execute(
            "SELECT status, error_message FROM storage_schema_migrations "
            "WHERE migration_id = 'core-002'"
        ).fetchone()
    assert row is not None and row[0] == "FAILED" and "RuntimeError" in row[1]


def test_migrated_modules_have_no_direct_duckdb_connect() -> None:
    root = Path(__file__).parents[3] / "src/qmt_agent_trader/data"
    migrated = [
        root / "storage.py",
        root / "providers/tushare/quota.py",
        root / "providers/tushare/ledger_migration.py",
    ]
    assert all("duckdb.connect" not in path.read_text() for path in migrated)
