from __future__ import annotations

import hashlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from qmt_agent_trader.persistence.artifacts import (
    ArtifactMetadata,
    ArtifactStore,
)
from qmt_agent_trader.persistence.atomic_files import AtomicFileStore
from qmt_agent_trader.persistence.errors import (
    StorageConflictError,
    StorageValidationError,
)
from qmt_agent_trader.persistence.locks import LockManager


@pytest.fixture
def store(tmp_path: Path) -> ArtifactStore:
    manager = LockManager(tmp_path / "locks")
    return ArtifactStore(tmp_path / "artifacts", AtomicFileStore(manager), manager)


def _metadata(artifact_id: str = "run_1") -> ArtifactMetadata:
    return ArtifactMetadata(
        artifact_id=artifact_id,
        artifact_type="research_report",
        producer="tests.artifact_store",
        related_run_id="run_1",
    )


def test_create_writes_exact_content_and_manifest_and_verifies_hash(
    store: ArtifactStore,
) -> None:
    content = b'{"run_id":"run_1"}\n'

    receipt = store.create("reports/run_1.json", content, metadata=_metadata())

    assert receipt.path.read_bytes() == content
    assert receipt.manifest.content_hash == hashlib.sha256(content).hexdigest()
    assert receipt.manifest.relative_path == "reports/run_1.json"
    assert receipt.manifest.related_run_id == "run_1"
    assert receipt.manifest.schema_version == 1
    assert receipt.manifest_path.exists()
    assert store.verify("run_1").verified is True


def test_same_artifact_id_or_target_name_cannot_overwrite(store: ArtifactStore) -> None:
    store.create("reports/run_1.json", b"one", metadata=_metadata())

    with pytest.raises(StorageConflictError):
        store.create("reports/run_2.json", b"two", metadata=_metadata())
    with pytest.raises(StorageConflictError):
        store.create("reports/run_1.json", b"two", metadata=_metadata("run_2"))

    assert store.path_for("reports/run_1.json").read_bytes() == b"one"


@pytest.mark.parametrize("relative_path", ["../escape.json", "/tmp/escape.json", "a/../../x"])
def test_create_rejects_path_traversal(
    store: ArtifactStore,
    relative_path: str,
) -> None:
    with pytest.raises(StorageValidationError):
        store.create(relative_path, b"payload", metadata=_metadata())


def test_fault_before_publish_leaves_no_official_artifact_or_manifest(
    tmp_path: Path,
) -> None:
    manager = LockManager(tmp_path / "locks")

    def explode(stage: str, _path: Path) -> None:
        if stage == "before_replace":
            raise OSError("injected")

    store = ArtifactStore(
        tmp_path / "artifacts",
        AtomicFileStore(manager),
        manager,
        fault_hook=explode,
    )

    with pytest.raises(Exception, match="atomic write failed"):
        store.create("reports/run_1.json", b"payload", metadata=_metadata())

    assert not store.path_for("reports/run_1.json").exists()
    assert not store.manifest_path_for("run_1").exists()


def test_fault_after_content_publish_rolls_back_and_never_reports_success(
    tmp_path: Path,
) -> None:
    manager = LockManager(tmp_path / "locks")

    def explode(stage: str, _path: Path) -> None:
        if stage == "after_content_publish":
            raise OSError("injected after content publish")

    store = ArtifactStore(
        tmp_path / "artifacts",
        AtomicFileStore(manager),
        manager,
        fault_hook=explode,
    )

    with pytest.raises(OSError, match="after content publish"):
        store.create("reports/run_1.json", b"payload", metadata=_metadata())

    assert not store.path_for("reports/run_1.json").exists()
    assert not store.manifest_path_for("run_1").exists()


def test_existing_manifest_identity_blocks_publication_even_when_content_is_missing(
    store: ArtifactStore,
) -> None:
    manifest_path = store.manifest_path_for("run_1")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text("{}", encoding="utf-8")

    with pytest.raises(StorageConflictError):
        store.create("reports/run_1.json", b"payload", metadata=_metadata())

    assert not store.path_for("reports/run_1.json").exists()


def test_verify_rejects_manifest_identity_or_relative_path_substitution(
    store: ArtifactStore,
) -> None:
    store.create("reports/run_1.json", b"one", metadata=_metadata("run_1"))
    manifest_path = store.manifest_path_for("run_1")
    payload = __import__("json").loads(manifest_path.read_text(encoding="utf-8"))
    payload["artifact_id"] = "attacker"
    manifest_path.write_text(__import__("json").dumps(payload), encoding="utf-8")

    with pytest.raises(StorageValidationError, match="identity"):
        store.verify("run_1", expected_relative_path="reports/run_1.json")

    payload["artifact_id"] = "run_1"
    manifest_path.write_text(__import__("json").dumps(payload), encoding="utf-8")
    with pytest.raises(StorageValidationError, match="relative path"):
        store.verify("run_1", expected_relative_path="reports/other.json")


def test_diagnostics_report_orphan_missing_content_and_hash_mismatch(
    store: ArtifactStore,
) -> None:
    store.create("reports/good.json", b"good", metadata=_metadata("good"))
    orphan = store.path_for("reports/orphan.json")
    orphan.parent.mkdir(parents=True, exist_ok=True)
    orphan.write_bytes(b"orphan")
    store.create("reports/missing.json", b"missing", metadata=_metadata("missing"))
    store.path_for("reports/missing.json").unlink()
    store.create("reports/tampered.json", b"original", metadata=_metadata("tampered"))
    store.path_for("reports/tampered.json").write_bytes(b"tampered")

    diagnostics = {(item.code, item.relative_path) for item in store.diagnose()}

    assert ("ORPHAN_ARTIFACT", "reports/orphan.json") in diagnostics
    assert ("MISSING_ARTIFACT", "reports/missing.json") in diagnostics
    assert ("HASH_MISMATCH", "reports/tampered.json") in diagnostics


def test_diagnostics_reject_manifest_filename_identity_substitution(
    store: ArtifactStore,
) -> None:
    receipt = store.create("reports/good.json", b"good", metadata=_metadata("good"))
    substituted = receipt.manifest_path.with_name("attacker.json")
    receipt.manifest_path.rename(substituted)

    diagnostics = {(item.code, item.relative_path) for item in store.diagnose()}

    assert ("INVALID_MANIFEST", substituted.relative_to(store.root).as_posix()) in diagnostics
    assert ("ORPHAN_ARTIFACT", "reports/good.json") in diagnostics


def test_concurrent_create_has_exactly_one_winner(store: ArtifactStore) -> None:
    def create(content: bytes) -> str:
        try:
            store.create("reports/run_1.json", content, metadata=_metadata())
        except StorageConflictError:
            return "conflict"
        return "created"

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(create, (b"first", b"second")))

    assert sorted(results) == ["conflict", "created"]
    assert store.verify("run_1").verified is True
