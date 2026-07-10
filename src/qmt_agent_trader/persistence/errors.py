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
