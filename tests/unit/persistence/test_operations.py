from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import pytest

from qmt_agent_trader.core.config import Settings
from qmt_agent_trader.persistence.errors import StorageBackupError, StorageValidationError
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
