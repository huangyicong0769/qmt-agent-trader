"""Secret-safe structured persistence errors."""

from __future__ import annotations

from pathlib import Path


class StorageError(RuntimeError):
    def __init__(
        self,
        *,
        store_name: str,
        path: Path | None = None,
        database_path: Path | None = None,
        operation: str,
        reason: str,
        recoverable: bool = False,
        suggested_repair: str | None = None,
        original_error: BaseException | None = None,
    ) -> None:
        selected_path = path or database_path
        self.store_name = store_name
        self.path = selected_path.expanduser().resolve() if selected_path is not None else None
        self.database_path = (
            database_path.expanduser().resolve() if database_path is not None else None
        )
        self.operation = operation
        self.reason = reason
        self.recoverable = recoverable
        self.suggested_repair = suggested_repair
        self.original_error_type = (
            type(original_error).__name__ if original_error is not None else None
        )
        message = f"{store_name} storage operation {operation} failed: {reason}"
        if self.path is not None:
            message += f" ({self.path})"
        super().__init__(message)
        if original_error is not None:
            self.__cause__ = original_error


class StorageUnavailableError(StorageError): ...
class StorageCorruptError(StorageError): ...
class StorageSchemaMismatchError(StorageError): ...
class StorageMigrationRequiredError(StorageError): ...
class StorageMigrationFailedError(StorageError): ...
class StorageLockTimeoutError(StorageError): ...
class StorageConflictError(StorageError): ...
class StorageRevisionConflictError(StorageConflictError): ...
class StorageValidationError(StorageError): ...
class StorageBackupError(StorageError): ...


class StorageAppendRollbackError(StorageError):
    def __init__(
        self,
        *,
        path: Path,
        append_error: BaseException,
        rollback_error: BaseException,
    ) -> None:
        self.original_append_error_type = type(append_error).__name__
        self.rollback_error_type = type(rollback_error).__name__
        super().__init__(
            store_name="atomic_files",
            path=path,
            operation="append_jsonl",
            reason=(
                "append and rollback failed "
                f"(append={self.original_append_error_type}, "
                f"rollback={self.rollback_error_type})"
            ),
            recoverable=False,
            suggested_repair="quarantine and inspect the JSONL stream",
            original_error=rollback_error,
        )
