"""Validated same-directory atomic file operations."""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any
from uuid import uuid4

import pandas as pd
import pyarrow.parquet as pq
import yaml
from pydantic import BaseModel, ValidationError

from qmt_agent_trader.persistence.errors import (
    StorageAppendRollbackError,
    StorageConflictError,
    StorageError,
    StorageValidationError,
)
from qmt_agent_trader.persistence.locks import LockManager

Validator = Callable[[Any], bool | None]
FaultHook = Callable[[str, Path], None]


class AtomicFileStore:
    def __init__(self, lock_manager: LockManager) -> None:
        self.lock_manager = lock_manager

    def write_bytes(
        self,
        path: Path,
        content: bytes,
        *,
        create_only: bool = False,
        validator: Callable[[bytes], bool | None] | None = None,
        fault_hook: FaultHook | None = None,
    ) -> None:
        with self.lock_manager.resource_lock(path):
            self.write_bytes_assume_locked(
                path,
                content,
                create_only=create_only,
                validator=validator,
                fault_hook=fault_hook,
            )

    def write_bytes_assume_locked(
        self,
        path: Path,
        content: bytes,
        *,
        create_only: bool = False,
        validator: Callable[[bytes], bool | None] | None = None,
        fault_hook: FaultHook | None = None,
    ) -> None:
        self._write(
            path, content, create_only=create_only, validator=validator, fault_hook=fault_hook
        )

    def write_text(self, path: Path, content: str, *, create_only: bool = False) -> None:
        self.write_bytes(path, content.encode("utf-8"), create_only=create_only)

    def write_json(
        self,
        path: Path,
        value: Any,
        *,
        create_only: bool = False,
        validator: Validator | None = None,
        model: type[BaseModel] | None = None,
        fault_hook: FaultHook | None = None,
    ) -> None:
        with self.lock_manager.resource_lock(path):
            self.write_json_assume_locked(
                path,
                value,
                create_only=create_only,
                validator=validator,
                model=model,
                fault_hook=fault_hook,
            )

    def write_json_assume_locked(
        self,
        path: Path,
        value: Any,
        *,
        create_only: bool = False,
        validator: Validator | None = None,
        model: type[BaseModel] | None = None,
        fault_hook: FaultHook | None = None,
    ) -> None:
        value = _model_value(value, model, path, "write_json")
        _validate(value, validator, path, "write_json")
        content = json.dumps(value, ensure_ascii=True, sort_keys=True, indent=2).encode() + b"\n"
        self._write(
            path,
            content,
            create_only=create_only,
            validator=lambda raw: _validate_json(raw, validator, model),
            fault_hook=fault_hook,
        )

    def write_yaml(
        self,
        path: Path,
        value: Any,
        *,
        create_only: bool = False,
        validator: Validator | None = None,
        model: type[BaseModel] | None = None,
    ) -> None:
        with self.lock_manager.resource_lock(path):
            self.write_yaml_assume_locked(
                path,
                value,
                create_only=create_only,
                validator=validator,
                model=model,
            )

    def write_yaml_assume_locked(
        self,
        path: Path,
        value: Any,
        *,
        create_only: bool = False,
        validator: Validator | None = None,
        model: type[BaseModel] | None = None,
    ) -> None:
        value = _model_value(value, model, path, "write_yaml")
        _validate(value, validator, path, "write_yaml")
        content = yaml.safe_dump(value, sort_keys=True).encode()
        self._write(
            path,
            content,
            create_only=create_only,
            validator=lambda raw: _validate_yaml(raw, validator, model),
        )

    def write_parquet(self, path: Path, frame: pd.DataFrame) -> None:
        with self.lock_manager.resource_lock(path):
            self.write_parquet_assume_locked(path, frame)

    def write_parquet_assume_locked(self, path: Path, frame: pd.DataFrame) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = _temp_path(path)
        try:
            frame.to_parquet(temp, index=False)
            parquet = pq.ParquetFile(temp)  # type: ignore[no-untyped-call]
            for index in range(parquet.num_row_groups):
                parquet.read_row_group(index)  # type: ignore[no-untyped-call]
            with temp.open("rb") as handle:
                os.fsync(handle.fileno())
            os.replace(temp, path)
            _fsync_directory(path.parent)
        except StorageError:
            raise
        except Exception as exc:
            raise StorageError(
                store_name="atomic_files",
                path=path,
                operation="write_parquet",
                reason="atomic write failed",
                original_error=exc,
            ) from exc
        finally:
            temp.unlink(missing_ok=True)

    def update_json(
        self, path: Path, update: Callable[[Any], Any], *, validator: Validator | None = None
    ) -> Any:
        with self.lock_manager.resource_lock(path):
            current = json.loads(path.read_text()) if path.exists() else None
            result = update(current)
            self.write_json_assume_locked(path, result, validator=validator)
            return result

    def append_jsonl(self, path: Path, record: Any, *, fsync: bool = True) -> None:
        encoded = _encode_jsonl(record, compact=True)
        if b"\n" in encoded[:-1]:
            raise StorageValidationError(
                store_name="atomic_files",
                path=path,
                operation="append_jsonl",
                reason="record encoded to more than one line",
            )
        with self.lock_manager.resource_lock(path):
            self._append_encoded_jsonl(path, encoded, fsync=fsync)

    def append_jsonl_assume_locked(self, path: Path, record: Any, *, fsync: bool = True) -> None:
        encoded = _encode_jsonl(record, compact=True)
        if b"\n" in encoded[:-1]:
            raise StorageValidationError(
                store_name="atomic_files",
                path=path,
                operation="append_jsonl",
                reason="record encoded to more than one line",
            )
        self._append_encoded_jsonl(path, encoded, fsync=fsync)

    def rotate_and_append_jsonl(
        self,
        path: Path,
        record: Any,
        *,
        rotation_bytes: int | None = None,
        fsync: bool = True,
        compact: bool = True,
    ) -> None:
        encoded = _encode_jsonl(record, compact=compact)
        with self.lock_manager.resource_lock(path):
            if (
                rotation_bytes is not None
                and path.exists()
                and path.stat().st_size > 0
                and path.stat().st_size + len(encoded) > rotation_bytes
            ):
                rotated = _next_jsonl_generation(path)
                os.replace(path, rotated)
                _fsync_directory(path.parent)
            self._append_encoded_jsonl(path, encoded, fsync=fsync)

    def _append_encoded_jsonl(self, path: Path, encoded: bytes, *, fsync: bool) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o600)
        original_length = os.fstat(descriptor).st_size
        try:
            written = os.write(descriptor, encoded)
            if written != len(encoded):
                raise OSError("partial JSONL append")
            if fsync:
                os.fsync(descriptor)
        except Exception as exc:
            try:
                os.ftruncate(descriptor, original_length)
                if fsync:
                    os.fsync(descriptor)
            except Exception as rollback_exc:
                raise StorageAppendRollbackError(
                    path=path,
                    append_error=exc,
                    rollback_error=rollback_exc,
                ) from rollback_exc
            raise StorageError(
                store_name="atomic_files",
                path=path,
                operation="append_jsonl",
                reason="append failed and original length was restored",
                recoverable=True,
                original_error=exc,
            ) from exc
        finally:
            os.close(descriptor)

    def _write(
        self,
        path: Path,
        content: bytes,
        *,
        create_only: bool,
        validator: Callable[[bytes], bool | None] | None = None,
        fault_hook: FaultHook | None = None,
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        if create_only and path.exists():
            raise StorageConflictError(
                store_name="atomic_files",
                path=path,
                operation="create",
                reason="target already exists",
            )
        temp = _temp_path(path)
        try:
            descriptor = os.open(temp, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            try:
                with os.fdopen(descriptor, "wb", closefd=False) as handle:
                    handle.write(content)
                    handle.flush()
                    os.fsync(handle.fileno())
            finally:
                os.close(descriptor)
            if validator is not None and validator(temp.read_bytes()) is False:
                raise StorageValidationError(
                    store_name="atomic_files",
                    path=path,
                    operation="validate",
                    reason="write validation failed",
                )
            if fault_hook is not None:
                fault_hook("before_replace", temp)
            if create_only:
                try:
                    os.link(temp, path)
                except FileExistsError as exc:
                    raise StorageConflictError(
                        store_name="atomic_files",
                        path=path,
                        operation="create",
                        reason="target already exists",
                        original_error=exc,
                    ) from exc
            else:
                os.replace(temp, path)
            _fsync_directory(path.parent)
        except StorageError:
            raise
        except Exception as exc:
            raise StorageError(
                store_name="atomic_files",
                path=path,
                operation="write",
                reason="atomic write failed",
                original_error=exc,
            ) from exc
        finally:
            temp.unlink(missing_ok=True)


def _validate(value: Any, validator: Validator | None, path: Path, operation: str) -> None:
    try:
        valid = True if validator is None else validator(value)
    except Exception as exc:
        raise StorageValidationError(
            store_name="atomic_files",
            path=path,
            operation=operation,
            reason="schema validation failed",
            original_error=exc,
        ) from exc
    if valid is False:
        raise StorageValidationError(
            store_name="atomic_files",
            path=path,
            operation=operation,
            reason="schema validation failed",
        )


def _validate_json(raw: bytes, validator: Validator | None, model: type[BaseModel] | None) -> bool:
    value = json.loads(raw)
    if model is not None:
        model.model_validate(value)
    return validator is None or validator(value) is not False


def _validate_yaml(raw: bytes, validator: Validator | None, model: type[BaseModel] | None) -> bool:
    value = yaml.safe_load(raw)
    if model is not None:
        model.model_validate(value)
    return validator is None or validator(value) is not False


def _model_value(value: Any, model: type[BaseModel] | None, path: Path, operation: str) -> Any:
    try:
        if model is not None:
            return model.model_validate(value).model_dump(mode="json")
        if isinstance(value, BaseModel):
            return value.model_dump(mode="json")
        return value
    except ValidationError as exc:
        raise StorageValidationError(
            store_name="atomic_files",
            path=path,
            operation=operation,
            reason="model validation failed",
            original_error=exc,
        ) from exc


def _temp_path(path: Path) -> Path:
    return path.with_name(f".{path.name}.{uuid4().hex}.tmp")


def _encode_jsonl(record: Any, *, compact: bool) -> bytes:
    separators = (",", ":") if compact else None
    return json.dumps(record, ensure_ascii=True, separators=separators).encode() + b"\n"


def _next_jsonl_generation(path: Path) -> Path:
    prefix = f"{path.name}."
    generations = [
        int(candidate.name.removeprefix(prefix))
        for candidate in path.parent.glob(f"{path.name}.*")
        if candidate.name.removeprefix(prefix).isdigit()
    ]
    return path.with_name(f"{path.name}.{max(generations, default=0) + 1:012d}")


def _fsync_directory(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)
