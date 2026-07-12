from __future__ import annotations

import json
import shutil
import threading
import time
from contextlib import contextmanager
from pathlib import Path

import pandas as pd
import pytest

from qmt_agent_trader.agent.sandbox import CodeSandbox
from qmt_agent_trader.backtest.service import BacktestRunSummary, run_backtest_report
from qmt_agent_trader.core.config import Settings
from qmt_agent_trader.data.storage import DataLake
from qmt_agent_trader.persistence.artifacts import ArtifactMetadata, artifact_store_for_root
from qmt_agent_trader.persistence.database import DatabaseCoordinator
from qmt_agent_trader.persistence.errors import StorageBackupError, StorageValidationError
from qmt_agent_trader.persistence.health import storage_health_payload
from qmt_agent_trader.persistence.initialization import storage_migrations
from qmt_agent_trader.persistence.locks import LockManager
from qmt_agent_trader.persistence.migrations import MigrationRegistry
from qmt_agent_trader.persistence.operations import StorageOperations
from qmt_agent_trader.persistence.paths import PersistencePaths
from qmt_agent_trader.services.order_plan_service import (
    append_order_plan_event,
    build_sample_paper_order_plan,
    load_order_plan_events,
    save_order_plan,
)
from qmt_agent_trader.services.research_report_service import save_research_report
from qmt_agent_trader.web.chat_repository import ChatSessionRepository
from qmt_agent_trader.web.schemas import ChatSession


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


def test_verify_detects_corrupt_order_plan_event_stream(
    operations: StorageOperations,
) -> None:
    plan = build_sample_paper_order_plan("s1")
    plan_store = artifact_store_for_root(
        operations.paths.order_plans_root, lock_manager=operations.locks
    )
    save_order_plan(plan, artifact_store=plan_store)
    append_order_plan_event(
        plan.order_plan_id,
        event_type="RISK_CHECKED",
        actor="test",
        artifact_store=plan_store,
    )
    event_path = next((operations.paths.order_plans_root / ".events").glob("*.jsonl"))
    event_path.write_bytes(event_path.read_bytes() + b'{"broken"')

    result = operations.verify(deep=True)

    assert not result.healthy
    assert any(
        item.component == "order_plan_events" and item.code == "TRUNCATED_TAIL"
        for item in result.diagnostics
    )


def test_verify_reports_immutable_pending_migrations_without_mutation(
    operations: StorageOperations,
) -> None:
    migrations = storage_migrations()
    MigrationRegistry(operations.database).apply(migrations[:1])
    before = operations.paths.control_db_path.read_bytes()

    result = operations.verify()

    assert not result.healthy
    assert any(item.code == "MIGRATION_PENDING" for item in result.diagnostics)
    assert operations.paths.control_db_path.read_bytes() == before


def test_transient_report_cache_and_tool_payload_are_excluded_but_governed_reports_are_included(
    operations: StorageOperations,
) -> None:
    cache = operations.paths.cache_root / "valid.json"
    payload = operations.paths.reports_root / "tool_payloads/call.json"
    cache.parent.mkdir(parents=True)
    payload.parent.mkdir(parents=True)
    cache.write_text('{"cache": true}')
    payload.write_text('{"transport": true}')
    report_root = operations.paths.reports_root / "research"
    artifact_store_for_root(report_root, lock_manager=operations.locks).create(
        "research_run.json",
        b'{"governed": true}',
        metadata=ArtifactMetadata(
            artifact_id="research-run",
            artifact_type="research_report",
            producer="test",
        ),
    )

    assert operations.verify(deep=True).healthy
    receipt = operations.backup()
    manifest = json.loads(receipt.manifest_path.read_text())
    sources = {item["source"] for item in manifest["files"]}
    assert "reports/research/research_run.json" in sources
    assert not any("reports/cache" in item or "reports/tool_payloads" in item for item in sources)


def test_composed_generated_code_root_is_verified_and_backed_up(
    operations: StorageOperations,
) -> None:
    generated = operations.paths.project_root / "src/qmt_agent_trader/agent/generated"
    sandbox = CodeSandbox(generated_root=generated, lock_manager=operations.locks)
    sandbox.write_candidate_file(
        "factors/factor.py", "# governed candidate", artifact_id="factor-candidate"
    )

    assert operations.catalog.by_name("generated_code").path == sandbox.generated_root
    assert operations.verify(deep=True).healthy
    manifest = json.loads(operations.backup().manifest_path.read_text())
    assert any(
        item["source"] == "src/qmt_agent_trader/agent/generated/factors/factor.py"
        for item in manifest["files"]
    )


def test_governed_store_is_diagnosed_once_per_verify(
    operations: StorageOperations, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = operations.paths.approvals_root
    store = artifact_store_for_root(root, lock_manager=operations.locks)
    for index in range(3):
        store.create(
            f"approval-{index}.yaml",
            b"status: APPROVED\n",
            metadata=ArtifactMetadata(
                artifact_id=f"approval-{index}", artifact_type="approval", producer="test"
            ),
        )
    calls = 0
    original = type(store).diagnose_assume_locked

    def counted(self: object) -> list[object]:
        nonlocal calls
        calls += 1
        return original(self)  # type: ignore[arg-type, return-value]

    monkeypatch.setattr(type(store), "diagnose_assume_locked", counted)

    expected = sum(
        1
        for definition in operations.catalog.stores
        if definition.governed and definition.path.is_dir()
    )
    assert operations.verify(deep=True).healthy
    assert calls == expected


def test_backup_excludes_cache_temp_and_locks_and_verifies_hashes(
    operations: StorageOperations,
) -> None:
    repository = ChatSessionRepository(
        operations.paths.sessions_root,
        locks_root=operations.paths.locks_root,
        quarantine_root=operations.paths.quarantine_root / "sessions",
    )
    repository.create(ChatSession(session_id="s"))
    operations.paths.cache_root.mkdir(parents=True)
    (operations.paths.cache_root / "skip.json").write_text("cache")
    operations.paths.locks_root.mkdir(parents=True, exist_ok=True)
    (operations.paths.locks_root / "active.lock").write_text("")
    (operations.paths.data_root / "orphan.tmp").write_text("temp")

    receipt = operations.backup()

    manifest = json.loads(receipt.manifest_path.read_text())
    paths = {item["source"] for item in manifest["files"]}
    assert "sessions/s.json" in paths
    assert not any("cache" in item or item.endswith(".tmp") or "locks" in item for item in paths)
    assert operations.verify_backup(receipt.path).healthy


def test_backup_waits_for_active_writer_barrier(operations: StorageOperations) -> None:
    record = operations.paths.lake_root / "raw/state.json"
    record.parent.mkdir(parents=True)
    record.write_text('{"value": "before"}')
    acquired = threading.Event()

    def writer() -> None:
        with operations.locks.resource_lock(record):
            acquired.set()
            time.sleep(0.15)
            record.write_text('{"value": "after"}')

    thread = threading.Thread(target=writer)
    thread.start()
    acquired.wait(timeout=1)
    started = time.monotonic()
    receipt = operations.backup()
    elapsed = time.monotonic() - started
    thread.join()

    assert elapsed >= 0.1
    backed_up = receipt.path / "files" / record.relative_to(operations.paths.project_root)
    assert json.loads(backed_up.read_text()) == {"value": "after"}


def test_report_artifact_creation_cannot_appear_mid_backup(
    operations: StorageOperations, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = operations.paths.reports_root / "research"
    store = artifact_store_for_root(root, lock_manager=operations.locks)
    store.create(
        "first.json",
        b'{"generation": 1}',
        metadata=ArtifactMetadata(
            artifact_id="first", artifact_type="research_report", producer="test"
        ),
    )
    entered, release, writer_done = threading.Event(), threading.Event(), threading.Event()
    original_copy = shutil.copy2

    def blocking_copy(source: Path, target: Path) -> Path:
        if source.name == "first.json":
            entered.set()
            release.wait(timeout=2)
        return original_copy(source, target)

    monkeypatch.setattr("qmt_agent_trader.persistence.operations.shutil.copy2", blocking_copy)
    receipts: list[object] = []
    backup_thread = threading.Thread(target=lambda: receipts.append(operations.backup()))
    backup_thread.start()
    assert entered.wait(timeout=2)

    def create_second() -> None:
        store.create(
            "second.json",
            b'{"generation": 2}',
            metadata=ArtifactMetadata(
                artifact_id="second", artifact_type="research_report", producer="test"
            ),
        )
        writer_done.set()

    writer = threading.Thread(target=create_second)
    writer.start()
    assert not writer_done.wait(timeout=0.1)
    release.set()
    backup_thread.join(timeout=3)
    writer.join(timeout=3)
    assert writer_done.is_set()
    receipt = receipts[0]
    manifest = json.loads(receipt.manifest_path.read_text())
    assert not any(item["source"].endswith("second.json") for item in manifest["files"])


@pytest.mark.parametrize("writer_kind", ["backtest", "research"])
def test_real_report_writer_uses_custom_settings_backup_barrier(
    operations: StorageOperations,
    monkeypatch: pytest.MonkeyPatch,
    writer_kind: str,
) -> None:
    lake = DataLake(
        operations.paths.lake_root,
        operations.paths.control_db_path,
        lock_manager=operations.locks,
        database_coordinator=operations.database,
    )
    monkeypatch.setattr(
        "qmt_agent_trader.backtest.service.run_single_symbol_backtest",
        lambda *_args, **_kwargs: BacktestRunSummary(
            run_id="bt_barrier",
            symbol="000001.SZ",
            signal_date="20260101",
            quantity=100,
            fills=0,
            execution_dates=[],
            leakage_valid=True,
        ),
    )
    done = threading.Event()

    def write_report() -> None:
        if writer_kind == "backtest":
            run_backtest_report(lake, reports_dir=operations.paths.reports_root / "backtests")
        else:
            save_research_report(
                operations.paths.reports_root / "research",
                artifact_type="test",
                title="barrier",
                payload={},
                lock_manager=operations.locks,
            )
        done.set()

    with operations.locks.backup_barrier():
        writer = threading.Thread(target=write_report)
        writer.start()
        assert not done.wait(timeout=0.1)
    writer.join(timeout=2)
    assert done.is_set()


def test_nonincremental_parquet_write_waits_for_backup_barrier(
    operations: StorageOperations, monkeypatch: pytest.MonkeyPatch
) -> None:
    lake = DataLake(
        operations.paths.lake_root,
        operations.paths.control_db_path,
        lock_manager=operations.locks,
        database_coordinator=operations.database,
    )
    lake.write_parquet(pd.DataFrame({"generation": [1]}), "raw", "sample")
    entered, release, writer_done = threading.Event(), threading.Event(), threading.Event()
    original_copy = shutil.copy2

    def blocking_copy(source: Path, target: Path) -> Path:
        if source.name == "sample.parquet":
            entered.set()
            release.wait(timeout=2)
        return original_copy(source, target)

    monkeypatch.setattr("qmt_agent_trader.persistence.operations.shutil.copy2", blocking_copy)
    receipts: list[object] = []
    backup_thread = threading.Thread(target=lambda: receipts.append(operations.backup()))
    backup_thread.start()
    assert entered.wait(timeout=2)
    writer = threading.Thread(
        target=lambda: (
            lake.write_parquet(pd.DataFrame({"generation": [2]}), "raw", "sample"),
            writer_done.set(),
        )
    )
    writer.start()
    assert not writer_done.wait(timeout=0.1)
    release.set()
    backup_thread.join(timeout=3)
    writer.join(timeout=3)
    receipt = receipts[0]
    copied = receipt.path / "files/data/lake/raw/sample.parquet"
    assert pd.read_parquet(copied)["generation"].tolist() == [1]
    assert pd.read_parquet(lake.dataset_path("raw", "sample"))["generation"].tolist() == [2]


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


@pytest.mark.parametrize("kind", ["parquet", "artifact"])
def test_corrupt_source_snapshot_cannot_publish_success(
    operations: StorageOperations, kind: str
) -> None:
    if kind == "parquet":
        source = operations.paths.lake_root / "raw/broken.parquet"
        source.parent.mkdir(parents=True)
        source.write_bytes(b"not-parquet")
    else:
        store = artifact_store_for_root(
            operations.paths.reports_root / "research", lock_manager=operations.locks
        )
        receipt = store.create(
            "broken.json",
            b'{"valid": true}',
            metadata=ArtifactMetadata(
                artifact_id="broken", artifact_type="research_report", producer="test"
            ),
        )
        receipt.path.write_bytes(b"tampered")

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
    MigrationRegistry(operations.database).apply(storage_migrations())
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


def test_verify_and_quarantine_share_versioned_record_hash_validation(
    operations: StorageOperations,
) -> None:
    repository = ChatSessionRepository(
        operations.paths.sessions_root,
        locks_root=operations.paths.locks_root,
        quarantine_root=operations.paths.quarantine_root / "sessions",
    )
    repository.create(ChatSession(session_id="tampered"))
    record = operations.paths.sessions_root / "tampered.json"
    payload = json.loads(record.read_text())
    payload["content_hash"] = "0" * 64
    record.write_text(json.dumps(payload))

    verification = operations.verify()
    assert any(
        diagnostic.component == "sessions" and diagnostic.code == "HASH_MISMATCH"
        for diagnostic in verification.diagnostics
    )

    receipt = operations.quarantine("sessions", record.name)
    assert receipt.path.exists()


def test_verify_and_quarantine_share_versioned_record_model_validation(
    operations: StorageOperations,
) -> None:
    repository = ChatSessionRepository(
        operations.paths.sessions_root,
        locks_root=operations.paths.locks_root,
        quarantine_root=operations.paths.quarantine_root / "sessions",
    )
    repository.create(ChatSession(session_id="invalid-model"))
    record = operations.paths.sessions_root / "invalid-model.json"
    payload = json.loads(record.read_text())
    payload["messages"] = "not-a-list"
    payload["content_hash"] = repository.records._hash(
        {key: value for key, value in payload.items() if key != "content_hash"}
    )
    record.write_text(json.dumps(payload))

    verification = operations.verify()
    assert any(
        diagnostic.component == "sessions" and diagnostic.code == "INVALID_CONTENT"
        for diagnostic in verification.diagnostics
    )
    assert operations.quarantine("sessions", record.name).path.exists()


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


def test_quarantine_accepts_unmanifested_governed_artifact(
    operations: StorageOperations,
) -> None:
    legacy = operations.paths.approvals_root / "legacy.approval.yaml"
    legacy.parent.mkdir(parents=True)
    legacy.write_text("status: APPROVED\n")

    receipt = operations.quarantine("approvals", legacy.name)

    assert receipt.path.exists()
    assert not legacy.exists()


def test_quarantine_accepts_manifest_hash_corrupt_artifact(
    operations: StorageOperations,
) -> None:
    root = operations.paths.approvals_root
    store = artifact_store_for_root(root, lock_manager=operations.locks)
    receipt = store.create(
        "bad.approval.yaml",
        b"status: APPROVED\n",
        metadata=ArtifactMetadata(
            artifact_id="bad-approval", artifact_type="approval", producer="test"
        ),
    )
    receipt.path.write_text("status: TAMPERED\n")

    quarantined = operations.quarantine("approvals", "bad.approval.yaml")

    assert quarantined.path.exists()
    assert not receipt.path.exists()
    assert not receipt.manifest_path.exists()
    assert (quarantined.manifest_path.parent / "manifest.json").is_file()


def test_order_plan_quarantine_moves_manifest_content_and_events(
    operations: StorageOperations,
) -> None:
    plan = build_sample_paper_order_plan("s1")
    plan_store = artifact_store_for_root(
        operations.paths.order_plans_root, lock_manager=operations.locks
    )
    content_path = save_order_plan(plan, artifact_store=plan_store)
    append_order_plan_event(
        plan.order_plan_id,
        event_type="RISK_CHECKED",
        actor="test",
        artifact_store=plan_store,
    )
    event_path = next((operations.paths.order_plans_root / ".events").glob("*.jsonl"))
    content_path.write_bytes(b"tampered")

    quarantined = operations.quarantine("order_plans", content_path.name)

    unit_root = quarantined.manifest_path.parent
    assert not content_path.exists()
    assert not event_path.exists()
    assert (unit_root / "manifest.json").is_file()
    assert (unit_root / "auxiliary" / ".events" / event_path.name).is_file()


def test_governed_quarantine_accepts_missing_content_path(
    operations: StorageOperations,
) -> None:
    root = operations.paths.approvals_root
    store = artifact_store_for_root(root, lock_manager=operations.locks)
    created = store.create(
        "missing.yaml",
        b"status: APPROVED\n",
        metadata=ArtifactMetadata(
            artifact_id="missing-approval", artifact_type="approval", producer="test"
        ),
    )
    created.path.unlink()

    receipt = operations.quarantine("approvals", "missing.yaml")

    assert receipt.path.name == "manifest.json"
    assert not created.manifest_path.exists()


def test_event_only_quarantine_uses_order_plan_artifact_root_lock(
    operations: StorageOperations, monkeypatch: pytest.MonkeyPatch
) -> None:
    event_root = operations.paths.order_plans_root / ".events"
    event_path = event_root / ("0" * 64 + ".jsonl")
    event_path.parent.mkdir(parents=True)
    event_path.write_bytes(b'{"broken"')
    resources: list[str] = []
    original = operations.locks.resource_lock

    @contextmanager
    def recording(resource):
        resources.append(str(resource))
        with original(resource) as lock:
            yield lock

    monkeypatch.setattr(operations.locks, "resource_lock", recording)

    operations.quarantine("order_plan_events", event_path.name)

    assert f"artifact-store:{operations.paths.order_plans_root.resolve()}" in resources


def test_event_store_verify_and_append_share_one_root_snapshot(
    operations: StorageOperations, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = build_sample_paper_order_plan("s1")
    store = artifact_store_for_root(
        operations.paths.order_plans_root, lock_manager=operations.locks
    )
    save_order_plan(plan, artifact_store=store)
    append_order_plan_event(
        plan.order_plan_id,
        event_type="RISK_CHECKED",
        actor="test",
        artifact_store=store,
    )
    entered = threading.Event()
    release = threading.Event()
    append_attempting = threading.Event()
    append_done = threading.Event()
    verify_results = []
    from qmt_agent_trader.persistence import operations as operations_module

    original_verify = operations_module.verify_order_plan_event_stream

    def paused_verify(path, *, expected_order_plan_id=None):
        entered.set()
        assert release.wait(timeout=5)
        return original_verify(path, expected_order_plan_id=expected_order_plan_id)

    monkeypatch.setattr(operations_module, "verify_order_plan_event_stream", paused_verify)
    verify_thread = threading.Thread(
        target=lambda: verify_results.append(operations.verify(deep=True))
    )
    verify_thread.start()
    assert entered.wait(timeout=5)

    def append_during_verify() -> None:
        append_attempting.set()
        append_order_plan_event(
            plan.order_plan_id,
            event_type="PAPER_ACCEPTED",
            actor="test",
            artifact_store=store,
        )
        append_done.set()

    append_thread = threading.Thread(target=append_during_verify)
    append_thread.start()
    assert append_attempting.wait(timeout=5)
    assert not append_done.is_set()
    release.set()
    verify_thread.join(timeout=5)
    append_thread.join(timeout=5)

    assert verify_results and verify_results[0].healthy
    assert append_done.is_set()
    assert len(load_order_plan_events(plan.order_plan_id, artifact_store=store)) == 2


def test_shared_health_payload_recursively_scrubs_all_diagnostics() -> None:
    payload = storage_health_payload(
        component="cache",
        status="degraded",
        reason="degraded token=secret",
        warnings=["password=warning-secret"],
        repair_action="retry api_key=repair-secret",
    )
    assert set(payload) == {
        "storage_status",
        "storage_component",
        "storage_reason",
        "storage_warnings",
        "storage_repair_action",
    }
    assert "secret" not in json.dumps(payload).lower()


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
    assert report[str(known_path)]["active"] is False
    assert "stale" not in report[str(known_path)]
