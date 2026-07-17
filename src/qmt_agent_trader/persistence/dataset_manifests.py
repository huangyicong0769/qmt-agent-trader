"""Content manifests for governed Parquet datasets."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, ValidationError

from qmt_agent_trader.persistence.atomic_files import AtomicFileStore

_CONTENT_CHUNK_BYTES = 1024 * 1024


class DatasetContentManifest(BaseModel):
    schema_version: Literal["1"] = "1"
    dataset_name: str
    size_bytes: int = Field(ge=0)
    mtime_ns: int = Field(ge=0)
    ctime_ns: int = Field(ge=0)
    inode: int = Field(ge=0)
    sha256: str = Field(min_length=64, max_length=64)


def dataset_manifest_path(dataset_path: Path) -> Path:
    return dataset_path.with_suffix(dataset_path.suffix + ".manifest.json")


def ensure_dataset_content_fingerprint_assume_dataset_locked(
    path: Path,
    *,
    atomic_store: AtomicFileStore,
) -> str | None:
    if not path.exists():
        return None
    manifest_path = dataset_manifest_path(path)
    current = _stat_identity(path)
    manifest = _read_manifest(manifest_path)
    if manifest is not None and _matches_current_file(manifest, current):
        return manifest.sha256

    manifest = DatasetContentManifest(
        dataset_name=path.name,
        size_bytes=current["size_bytes"],
        mtime_ns=current["mtime_ns"],
        ctime_ns=current["ctime_ns"],
        inode=current["inode"],
        sha256=_content_digest(path),
    )
    atomic_store.write_json(manifest_path, manifest.model_dump(mode="json"))
    return manifest.sha256


def _read_manifest(path: Path) -> DatasetContentManifest | None:
    if not path.exists():
        return None
    try:
        return DatasetContentManifest.model_validate(
            json.loads(path.read_text(encoding="utf-8"))
        )
    except (OSError, ValueError, ValidationError):
        return None


def _stat_identity(path: Path) -> dict[str, int]:
    stat = path.stat()
    return {
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "ctime_ns": int(stat.st_ctime_ns),
        "inode": int(getattr(stat, "st_ino", 0)),
    }


def _matches_current_file(
    manifest: DatasetContentManifest,
    current: dict[str, int],
) -> bool:
    return (
        manifest.size_bytes == current["size_bytes"]
        and manifest.mtime_ns == current["mtime_ns"]
        and manifest.ctime_ns == current["ctime_ns"]
        and manifest.inode == current["inode"]
    )


def _content_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(_CONTENT_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()
