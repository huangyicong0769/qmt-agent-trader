"""Path-safe, create-only storage for governed immutable artifacts."""

from __future__ import annotations

import hashlib
import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from qmt_agent_trader.core.ids import shanghai_now_iso
from qmt_agent_trader.persistence.atomic_files import AtomicFileStore, FaultHook
from qmt_agent_trader.persistence.errors import (
    StorageConflictError,
    StorageValidationError,
)
from qmt_agent_trader.persistence.locks import LockManager


class ArtifactMetadata(BaseModel):
    """Caller-supplied immutable identity and provenance."""

    artifact_id: str = Field(min_length=1)
    artifact_type: str = Field(min_length=1)
    producer: str = Field(min_length=1)
    related_run_id: str | None = None
    related_strategy_id: str | None = None
    related_factor_id: str | None = None


class ArtifactManifest(ArtifactMetadata):
    schema_version: int = 1
    created_at: str
    content_hash: str
    byte_length: int = Field(ge=0)
    relative_path: str


class ArtifactReceipt(BaseModel):
    path: Path
    manifest_path: Path
    manifest: ArtifactManifest
    content: bytes


class ArtifactVerification(BaseModel):
    artifact_id: str
    verified: bool
    code: Literal["VERIFIED", "MISSING_ARTIFACT", "HASH_MISMATCH"]
    path: Path
    manifest_path: Path


class ArtifactDiagnostic(BaseModel):
    code: Literal[
        "ORPHAN_ARTIFACT",
        "MISSING_ARTIFACT",
        "HASH_MISMATCH",
        "INVALID_MANIFEST",
    ]
    relative_path: str
    artifact_id: str | None = None
    reason: str


@dataclass(frozen=True)
class ArtifactQuarantineReceipt:
    artifact_id: str
    original_content_path: Path
    original_manifest_path: Path
    quarantined_content_path: Path
    quarantined_manifest_path: Path
    sidecar_path: Path
    diagnostics: tuple[ArtifactDiagnostic, ...]
    quarantined_auxiliary_paths: tuple[Path, ...] = ()


class ArtifactStore:
    """Store exact artifact bytes and a deterministic manifest without replacement."""

    _MANIFEST_DIRECTORY = ".manifests"
    _AUXILIARY_DIRECTORIES = frozenset({".events"})

    def __init__(
        self,
        root: Path,
        atomic_store: AtomicFileStore,
        lock_manager: LockManager,
        *,
        fault_hook: FaultHook | None = None,
        now: Callable[[], str] = shanghai_now_iso,
    ) -> None:
        self.root = root.expanduser().resolve()
        self.atomic_store = atomic_store
        self.lock_manager = lock_manager
        self.fault_hook = fault_hook
        self.now = now

    def path_for(self, relative_path: str | Path) -> Path:
        raw = Path(relative_path)
        if raw.is_absolute() or self._MANIFEST_DIRECTORY in raw.parts:
            raise self._invalid_path(relative_path)
        candidate = (self.root / raw).resolve()
        if candidate == self.root or self.root not in candidate.parents:
            raise self._invalid_path(relative_path)
        return candidate

    def manifest_path_for(self, artifact_id: str) -> Path:
        if not artifact_id:
            raise StorageValidationError(
                store_name="artifacts",
                path=self.root,
                operation="manifest_path",
                reason="artifact_id must not be empty",
            )
        digest = hashlib.sha256(artifact_id.encode("utf-8")).hexdigest()
        return self.root / self._MANIFEST_DIRECTORY / f"{digest}.json"

    def create(
        self,
        relative_path: str | Path,
        content: bytes,
        *,
        metadata: ArtifactMetadata,
    ) -> ArtifactReceipt:
        path = self.path_for(relative_path)
        manifest_path = self.manifest_path_for(metadata.artifact_id)
        relative = path.relative_to(self.root).as_posix()
        manifest = ArtifactManifest(
            **metadata.model_dump(),
            created_at=self.now(),
            content_hash=hashlib.sha256(content).hexdigest(),
            byte_length=len(content),
            relative_path=relative,
        )
        resource = f"artifact-store:{self.root}"
        with self.lock_manager.resource_lock(resource):
            if path.exists() or manifest_path.exists():
                conflict_path = manifest_path if manifest_path.exists() else path
                raise StorageConflictError(
                    store_name="artifacts",
                    path=conflict_path,
                    operation="create",
                    reason="artifact identity or target path already exists",
                )
            created_content = False
            try:
                self.atomic_store.write_bytes_assume_locked(
                    path,
                    content,
                    create_only=True,
                    fault_hook=self.fault_hook,
                )
                created_content = True
                if self.fault_hook is not None:
                    self.fault_hook("after_content_publish", path)
                self.atomic_store.write_json_assume_locked(
                    manifest_path,
                    manifest,
                    create_only=True,
                    model=ArtifactManifest,
                )
            except Exception:
                if created_content and not manifest_path.exists():
                    path.unlink(missing_ok=True)
                raise
        return ArtifactReceipt(
            path=path, manifest_path=manifest_path, manifest=manifest, content=content
        )

    def load_manifest(self, artifact_id: str) -> ArtifactManifest:
        with self.lock_manager.resource_lock(self._resource):
            return self._load_manifest_assume_locked(artifact_id)

    def _load_manifest_assume_locked(self, artifact_id: str) -> ArtifactManifest:
        path = self.manifest_path_for(artifact_id)
        try:
            manifest = ArtifactManifest.model_validate_json(path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise StorageValidationError(
                store_name="artifacts",
                path=path,
                operation="load_manifest",
                reason="manifest is missing or invalid",
                original_error=exc,
            ) from exc
        if manifest.artifact_id != artifact_id:
            raise StorageValidationError(
                store_name="artifacts",
                path=path,
                operation="load_manifest",
                reason="manifest identity does not match requested artifact_id",
            )
        return manifest

    def verify(
        self,
        artifact_id: str,
        *,
        expected_relative_path: str | Path | None = None,
    ) -> ArtifactVerification:
        with self.lock_manager.resource_lock(self._resource):
            return self._verify_assume_locked(
                artifact_id, expected_relative_path=expected_relative_path
            )

    def _verify_assume_locked(
        self,
        artifact_id: str,
        *,
        expected_relative_path: str | Path | None = None,
    ) -> ArtifactVerification:
        manifest = self._validated_manifest_assume_locked(
            artifact_id, expected_relative_path=expected_relative_path
        )
        path = self.path_for(manifest.relative_path)
        manifest_path = self.manifest_path_for(artifact_id)
        if not path.is_file():
            return ArtifactVerification(
                artifact_id=artifact_id,
                verified=False,
                code="MISSING_ARTIFACT",
                path=path,
                manifest_path=manifest_path,
            )
        verified = hashlib.sha256(path.read_bytes()).hexdigest() == manifest.content_hash
        return ArtifactVerification(
            artifact_id=artifact_id,
            verified=verified,
            code="VERIFIED" if verified else "HASH_MISMATCH",
            path=path,
            manifest_path=manifest_path,
        )

    def read_verified(
        self,
        artifact_id: str,
        *,
        expected_relative_path: str | Path | None = None,
    ) -> bytes:
        with self.lock_manager.resource_lock(self._resource):
            manifest = self._validated_manifest_assume_locked(
                artifact_id,
                expected_relative_path=expected_relative_path,
            )
            path = self.path_for(manifest.relative_path)
            try:
                content = path.read_bytes()
            except OSError as exc:
                raise StorageValidationError(
                    store_name="artifacts",
                    path=path,
                    operation="read_verified",
                    reason="missing_artifact",
                    original_error=exc,
                ) from exc
            if hashlib.sha256(content).hexdigest() != manifest.content_hash:
                raise StorageValidationError(
                    store_name="artifacts",
                    path=path,
                    operation="read_verified",
                    reason="hash_mismatch",
                )
            return content

    def _validated_manifest_assume_locked(
        self,
        artifact_id: str,
        *,
        expected_relative_path: str | Path | None,
    ) -> ArtifactManifest:
        manifest = self._load_manifest_assume_locked(artifact_id)
        if expected_relative_path is not None:
            expected = self.path_for(expected_relative_path).relative_to(self.root).as_posix()
            if manifest.relative_path != expected:
                raise StorageValidationError(
                    store_name="artifacts",
                    path=self.manifest_path_for(artifact_id),
                    operation="verify_manifest_binding",
                    reason="manifest relative path does not match requested file",
                )
        return manifest

    def diagnose(self) -> list[ArtifactDiagnostic]:
        with self.lock_manager.resource_lock(self._resource):
            return self.diagnose_assume_locked()

    def diagnose_assume_locked(self) -> list[ArtifactDiagnostic]:
        diagnostics: list[ArtifactDiagnostic] = []
        referenced_paths: set[str] = set()
        manifest_root = self.root / self._MANIFEST_DIRECTORY
        for path in sorted(manifest_root.glob("*.json")):
            try:
                manifest = ArtifactManifest.model_validate_json(path.read_text(encoding="utf-8"))
                if self.manifest_path_for(manifest.artifact_id) != path:
                    raise ValueError("manifest filename does not bind artifact_id")
                if manifest.relative_path in referenced_paths:
                    raise ValueError("multiple manifests bind the same artifact path")
                artifact_path = self.path_for(manifest.relative_path)
            except Exception as exc:
                diagnostics.append(
                    ArtifactDiagnostic(
                        code="INVALID_MANIFEST",
                        relative_path=path.relative_to(self.root).as_posix(),
                        reason=type(exc).__name__,
                    )
                )
                continue
            referenced_paths.add(manifest.relative_path)
            if not artifact_path.is_file():
                diagnostics.append(
                    ArtifactDiagnostic(
                        code="MISSING_ARTIFACT",
                        relative_path=manifest.relative_path,
                        artifact_id=manifest.artifact_id,
                        reason="manifest references missing artifact",
                    )
                )
            elif hashlib.sha256(artifact_path.read_bytes()).hexdigest() != manifest.content_hash:
                diagnostics.append(
                    ArtifactDiagnostic(
                        code="HASH_MISMATCH",
                        relative_path=manifest.relative_path,
                        artifact_id=manifest.artifact_id,
                        reason="artifact bytes do not match manifest content_hash",
                    )
                )
        if self.root.exists():
            for path in sorted(self.root.rglob("*")):
                if (
                    not path.is_file()
                    or self._MANIFEST_DIRECTORY in path.parts
                    or any(part in self._AUXILIARY_DIRECTORIES for part in path.parts)
                ):
                    continue
                relative = path.relative_to(self.root).as_posix()
                if relative not in referenced_paths:
                    diagnostics.append(
                        ArtifactDiagnostic(
                            code="ORPHAN_ARTIFACT",
                            relative_path=relative,
                            reason="artifact has no manifest",
                        )
                    )
        return diagnostics

    def quarantine(
        self,
        *,
        artifact_id: str,
        expected_relative_path: str | Path | None = None,
        quarantine_root: Path,
        auxiliary_paths: tuple[Path, ...] = (),
    ) -> ArtifactQuarantineReceipt:
        """Move a corrupt governed artifact and its manifest as one recoverable unit."""
        with self.lock_manager.resource_lock(self._resource):
            manifest = self._validated_manifest_assume_locked(
                artifact_id, expected_relative_path=expected_relative_path
            )
            content_path = self.path_for(manifest.relative_path)
            manifest_path = self.manifest_path_for(artifact_id)
            diagnostics = tuple(
                item
                for item in self.diagnose_assume_locked()
                if item.artifact_id == artifact_id
                or item.relative_path == manifest.relative_path
            )
            if not diagnostics:
                raise StorageValidationError(
                    store_name="artifacts",
                    path=content_path,
                    operation="quarantine",
                    reason="authoritative artifact is healthy",
                )
            artifact_digest = hashlib.sha256(artifact_id.encode()).hexdigest()
            unit_name = f"{self.now().replace(':', '')}-{artifact_digest}"
            unit_root = quarantine_root.expanduser().resolve() / unit_name
            quarantined_content = unit_root / "content" / manifest.relative_path
            quarantined_manifest = unit_root / "manifest.json"
            sidecar = unit_root / "QUARANTINE.json"
            quarantined_content.parent.mkdir(parents=True, exist_ok=False)
            moved_content = False
            moved_manifest = False
            moved_auxiliary: list[tuple[Path, Path]] = []
            try:
                os.replace(content_path, quarantined_content)
                moved_content = True
                os.replace(manifest_path, quarantined_manifest)
                moved_manifest = True
                for auxiliary in auxiliary_paths:
                    resolved = auxiliary.expanduser().resolve()
                    if self.root not in resolved.parents or not resolved.is_file():
                        raise StorageValidationError(
                            store_name="artifacts",
                            path=resolved,
                            operation="quarantine",
                            reason="auxiliary path is outside artifact root or missing",
                        )
                    target = unit_root / "auxiliary" / resolved.relative_to(self.root)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    os.replace(resolved, target)
                    moved_auxiliary.append((resolved, target))
                self.atomic_store.write_json_assume_locked(
                    sidecar,
                    {
                        "schema_version": 1,
                        "artifact_id": artifact_id,
                        "original_content_path": str(content_path),
                        "original_manifest_path": str(manifest_path),
                        "content_sha256": hashlib.sha256(
                            quarantined_content.read_bytes()
                        ).hexdigest(),
                        "manifest_sha256": hashlib.sha256(
                            quarantined_manifest.read_bytes()
                        ).hexdigest(),
                        "auxiliary_paths": [
                            {
                                "original_path": str(original),
                                "quarantine_path": str(target),
                                "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
                            }
                            for original, target in moved_auxiliary
                        ],
                        "diagnostics": [item.model_dump(mode="json") for item in diagnostics],
                        "quarantined_at": self.now(),
                    },
                    create_only=True,
                )
            except Exception:
                sidecar.unlink(missing_ok=True)
                for original, target in reversed(moved_auxiliary):
                    if target.exists():
                        original.parent.mkdir(parents=True, exist_ok=True)
                        os.replace(target, original)
                if moved_manifest and quarantined_manifest.exists():
                    manifest_path.parent.mkdir(parents=True, exist_ok=True)
                    os.replace(quarantined_manifest, manifest_path)
                if moved_content and quarantined_content.exists():
                    content_path.parent.mkdir(parents=True, exist_ok=True)
                    os.replace(quarantined_content, content_path)
                raise
            return ArtifactQuarantineReceipt(
                artifact_id=artifact_id,
                original_content_path=content_path,
                original_manifest_path=manifest_path,
                quarantined_content_path=quarantined_content,
                quarantined_manifest_path=quarantined_manifest,
                sidecar_path=sidecar,
                diagnostics=diagnostics,
                quarantined_auxiliary_paths=tuple(target for _, target in moved_auxiliary),
            )

    @property
    def _resource(self) -> str:
        return f"artifact-store:{self.root}"

    def _invalid_path(self, relative_path: str | Path) -> StorageValidationError:
        return StorageValidationError(
            store_name="artifacts",
            path=self.root,
            operation="resolve_path",
            reason=f"unsafe artifact relative path: {relative_path}",
        )


def artifact_store_for_root(
    root: Path,
    *,
    lock_manager: LockManager | None = None,
) -> ArtifactStore:
    """Build the shared store for an explicitly injected domain root."""
    resolved_root = root.expanduser().resolve()
    manager = lock_manager or LockManager(resolved_root.parent / ".artifact-locks")
    return ArtifactStore(resolved_root, AtomicFileStore(manager), manager)
