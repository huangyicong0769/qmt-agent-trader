from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import pandas as pd
import pytest

from qmt_agent_trader.core.config import Settings
from qmt_agent_trader.persistence.database import DatabaseCoordinator
from qmt_agent_trader.persistence.errors import StorageBackupError, StorageValidationError
from qmt_agent_trader.persistence.locks import LockManager
from qmt_agent_trader.persistence.operations import StorageOperations
from qmt_agent_trader.persistence.paths import PersistencePaths


@pytest.fixture
def operations(tmp_path: Path) -> StorageOperations:
    paths = PersistencePaths.from_settings(Settings(project_root=tmp_path))
    return StorageOperations(paths)


def test_inventory_covers_every_canonical_path(operations: StorageOperations) -> None:
    names = {item.name for item in operations.inventory()}
    assert names == {store.name for store in operations.catalog.stores}
    assert all(
        item.owner and item.source_of_truth and item.lock_policy for item in operations.inventory()
    )


def test_verify_is_read_only_and_deep_detects_corrupt_parquet(
    operations: StorageOperations,
) -> None:
    target = operations.paths.lake_root / "raw/broken.parquet"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"PAR1broken-pagePAR1")
    before = {p: p.read_bytes() for p in operations.paths.project_root.rglob("*") if p.is_file()}

    result = operations.verify(deep=True)

    after = {p: p.read_bytes() for p in operations.paths.project_root.rglob("*") if p.is_file()}
    assert before == after
    assert not result.healthy
    assert any(d.code == "PARQUET_CORRUPT" for d in result.diagnostics)


def test_backup_excludes_cache_temp_and_locks_and_verifies_hashes(
    operations: StorageOperations,
) -> None:
    official = operations.paths.sessions_root / "s.json"
    official.parent.mkdir(parents=True)
    official.write_text('{"schema_version": 1}', encoding="utf-8")
    operations.paths.cache_root.mkdir(parents=True)
    (operations.paths.cache_root / "skip.json").write_text("cache")
    operations.paths.locks_root.mkdir(parents=True)
    (operations.paths.locks_root / "active.lock").write_text("")
    (operations.paths.data_root / "orphan.tmp").write_text("temp")

    receipt = operations.backup()

    manifest = json.loads(receipt.manifest_path.read_text())
    paths = {item["source"] for item in manifest["files"]}
    assert "sessions/s.json" in paths
    assert not any("cache" in item or item.endswith(".tmp") or "locks" in item for item in paths)
    assert operations.verify_backup(receipt.path).healthy


def test_backup_waits_for_active_writer_barrier(operations: StorageOperations) -> None:
    record = operations.paths.sessions_root / "s.json"
    record.parent.mkdir(parents=True)
    record.write_text("before")
    acquired = threading.Event()

    def writer() -> None:
        with operations.locks.resource_lock(record):
            acquired.set()
            time.sleep(0.15)
            record.write_text("after")

    thread = threading.Thread(target=writer)
    thread.start()
    acquired.wait(timeout=1)
    started = time.monotonic()
    receipt = operations.backup()
    elapsed = time.monotonic() - started
    thread.join()

    assert elapsed >= 0.1
    backed_up = receipt.path / "files" / "sessions/s.json"
    assert backed_up.read_text() == "after"


def test_backup_failure_has_no_success_marker(
    operations: StorageOperations, monkeypatch: pytest.MonkeyPatch
) -> None:
    record = operations.paths.sessions_root / "s.json"
    record.parent.mkdir(parents=True)
    record.write_text("official")
    monkeypatch.setattr(
        "qmt_agent_trader.persistence.operations.shutil.copy2",
        lambda *_: (_ for _ in ()).throw(OSError("injected")),
    )

    with pytest.raises(StorageBackupError):
        operations.backup()

    assert not list(operations.paths.backup_root.rglob("SUCCESS.json"))


def test_backup_verifier_rejects_manifest_traversal_and_extra_files(
    operations: StorageOperations,
) -> None:
    root = operations.paths.backup_root / "hostile"
    (root / "files").mkdir(parents=True)
    (root / "manifest.json").write_text(
        json.dumps(
            {"schema_version": 1, "files": [{"source": "../escape", "sha256": "x", "size": 1}]}
        )
    )
    (root / "files/extra").write_text("extra")

    result = operations.verify_backup(root)

    assert not result.healthy
    assert any(item.code in {"INVALID_MANIFEST", "EXTRA_FILE"} for item in result.diagnostics)


def test_backup_success_publish_failure_removes_final_directory(
    operations: StorageOperations, monkeypatch: pytest.MonkeyPatch
) -> None:
    record = operations.paths.sessions_root / "s.json"
    record.parent.mkdir(parents=True)
    record.write_text("official")
    original_write = operations.atomic.write_json

    def fail_success(path: Path, *args: object, **kwargs: object) -> None:
        if path.name == "SUCCESS.json":
            raise OSError("success marker injection")
        original_write(path, *args, **kwargs)

    monkeypatch.setattr(operations.atomic, "write_json", fail_success)
    with pytest.raises(StorageBackupError):
        operations.backup()

    assert not [
        path for path in operations.paths.backup_root.iterdir() if not path.name.startswith(".")
    ]


def test_backup_uses_coordinator_checkpoint_snapshot(
    operations: StorageOperations, monkeypatch: pytest.MonkeyPatch
) -> None:
    with operations.database.write_transaction("seed") as connection:
        connection.execute("CREATE TABLE snapshot_value(value INTEGER)")
        connection.execute("INSERT INTO snapshot_value VALUES (7)")
    called = False
    original = operations.database.checkpoint_copy

    def checkpoint_copy(target: Path) -> None:
        nonlocal called
        called = True
        original(target)

    monkeypatch.setattr(operations.database, "checkpoint_copy", checkpoint_copy)
    receipt = operations.backup()

    assert called
    copied = (
        receipt.path
        / "files"
        / operations.paths.control_db_path.relative_to(operations.paths.project_root)
    )
    coordinator = DatabaseCoordinator(copied, LockManager(operations.paths.locks_root / "read"))
    with coordinator.read_connection("verify snapshot", read_only=True) as connection:
        assert connection.execute("SELECT value FROM snapshot_value").fetchone() == (7,)


def test_quarantine_rejects_traversal_and_moves_invalid_record(
    operations: StorageOperations,
) -> None:
    with pytest.raises(StorageValidationError):
        operations.quarantine("sessions", "../secret")
    record = operations.paths.sessions_root / "bad.json"
    record.parent.mkdir(parents=True)
    record.write_text("{broken", encoding="utf-8")

    receipt = operations.quarantine("sessions", "bad.json")

    assert not record.exists()
    assert receipt.path.exists() and receipt.manifest_path.exists()


def test_quarantine_rejects_healthy_parquet(operations: StorageOperations) -> None:
    record = operations.paths.lake_root / "raw/healthy.parquet"
    record.parent.mkdir(parents=True)
    pd.DataFrame({"value": [1]}).to_parquet(record)

    with pytest.raises(StorageValidationError, match="valid"):
        operations.quarantine("lake_raw", "healthy.parquet")

    assert record.exists()


def test_quarantine_manifest_failure_rolls_source_back(
    operations: StorageOperations, monkeypatch: pytest.MonkeyPatch
) -> None:
    record = operations.paths.sessions_root / "bad.json"
    record.parent.mkdir(parents=True)
    original = b"{broken"
    record.write_bytes(original)

    def fail(*args: object, **kwargs: object) -> None:
        raise OSError("manifest injection")

    monkeypatch.setattr(operations.atomic, "write_json", fail)
    with pytest.raises(OSError, match="manifest injection"):
        operations.quarantine("sessions", "bad.json")

    assert record.read_bytes() == original
    assert not list((operations.paths.quarantine_root / "sessions").glob("*.quarantine"))


def test_health_payload_is_structured_and_secret_safe(operations: StorageOperations) -> None:
    payload = operations.health_payload(component="cache", reason="degraded token=secret")
    assert set(payload) == {
        "storage_status",
        "storage_component",
        "storage_reason",
        "storage_warnings",
        "storage_repair_action",
    }
    assert "secret" not in payload["storage_reason"]


def test_locks_report_maps_catalog_resources_and_marks_unknown(
    operations: StorageOperations,
) -> None:
    known = operations.catalog.by_name("sessions")
    known_path = operations.locks.lock_path_for_resource(known.lock_resource)
    known_path.parent.mkdir(parents=True)
    known_path.touch()
    unknown_path = operations.paths.locks_root / "resource-unknown.lock"
    unknown_path.touch()

    report = {item["path"]: item for item in operations.locks_report()}

    assert report[str(known_path)]["known_resource"] == "sessions"
    assert report[str(unknown_path)]["known_resource"] is None
    assert report[str(unknown_path)]["resource_status"] == "unknown"
    assert report[str(known_path)]["stale"] is False
