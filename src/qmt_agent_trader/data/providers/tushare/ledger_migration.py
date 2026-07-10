"""Migration and explicit recovery for the legacy Tushare usage Parquet ledger."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast
from uuid import uuid4

import pandas as pd
import pyarrow.parquet as pq

from qmt_agent_trader.data.providers.tushare.quota import (
    TushareUsageLedgerCorruptError,
    _record_to_row,
    _row_to_record,
    redact_params,
)

if TYPE_CHECKING:
    from qmt_agent_trader.data.providers.tushare.quota import TushareUsageLedger

_REQUIRED_COLUMNS = {
    "request_id",
    "run_id",
    "api_name",
    "params_hash",
    "params_redacted",
    "fields",
    "planned_at",
    "started_at",
    "finished_at",
    "status",
    "row_count",
    "error_type",
    "error_message",
    "token_hash",
    "execution_mode",
}


def migrate_legacy_usage_ledger(ledger: TushareUsageLedger) -> dict[str, Any]:
    if not ledger.path.exists():
        _finalize_archived_pending_migrations(ledger)
        return {"status": "NO_LEGACY_FILE", "imported_rows": 0, "skipped_rows": 0}
    with ledger.lock_manager.resource_lock(ledger.path):
        if not ledger.path.exists():
            return {"status": "NO_LEGACY_FILE", "imported_rows": 0, "skipped_rows": 0}
        frame = _read_complete_legacy(ledger.path)
        _validate_columns(ledger.path, frame)
        frame = _normalize_legacy_frame(ledger.path, frame)
        archive_path: Path
        migration_id: str
        imported = 0
        skipped = 0
        resume_pending = False
        with ledger.database_coordinator.write_transaction(
            "migrate_legacy_tushare_usage"
        ) as connection:
            pending = _pending_migration_for_source(connection, ledger.path)
            if pending is not None and not Path(str(pending[3])).exists():
                migration_id = str(pending[0])
                archive_path = Path(str(pending[3]))
                imported = int(pending[1])
                skipped = int(pending[2])
                resume_pending = True
            else:
                migration_id = str(uuid4())
                archive_path = _timestamped_path(
                    ledger.path.parent / "archive",
                    "migrated",
                )
            archive_path.parent.mkdir(parents=True, exist_ok=True)
            before = connection.execute(f"SELECT count(*) FROM {ledger.table_name}").fetchone()
            connection.register("legacy_tushare_usage", frame)
            connection.execute(
                f"""
                INSERT INTO {ledger.table_name}
                SELECT
                    request_id, run_id, api_name, params_hash,
                    params_redacted, fields, planned_at, started_at,
                    finished_at, status, row_count, error_type,
                    error_message, token_hash, execution_mode, current_timestamp
                FROM legacy_tushare_usage
                ON CONFLICT (request_id) DO NOTHING
                """
            )
            after = connection.execute(f"SELECT count(*) FROM {ledger.table_name}").fetchone()
            if before is None or after is None:
                raise RuntimeError("DuckDB did not return usage row counts during migration")
            if not resume_pending:
                imported = int(after[0]) - int(before[0])
                skipped = len(frame) - imported
                connection.execute(
                    """
                    INSERT INTO tushare_usage_migrations_v1
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        migration_id,
                        datetime.now(tz=UTC).replace(tzinfo=None),
                        str(ledger.path),
                        str(archive_path),
                        imported,
                        skipped,
                        "IMPORT_COMMITTED_PENDING_ARCHIVE",
                        None,
                    ],
                )
        os.replace(ledger.path, archive_path)
        _finalize_migration(ledger, migration_id)
        return {
            "status": "MIGRATED",
            "imported_rows": imported,
            "skipped_rows": skipped,
            "archive_path": str(archive_path),
        }


def repair_tushare_usage_ledger(
    ledger: TushareUsageLedger,
    *,
    quarantine_corrupt: bool,
) -> dict[str, Any]:
    if not ledger.path.exists():
        return {"status": "HEALTHY", "modified": False, "legacy_file": None}
    try:
        frame = _read_complete_legacy(ledger.path)
        _validate_columns(ledger.path, frame)
        frame = _normalize_legacy_frame(ledger.path, frame)
    except TushareUsageLedgerCorruptError as exc:
        if not quarantine_corrupt:
            return {
                "status": "CORRUPT",
                "modified": False,
                "path": str(exc.path),
                "error_type": exc.error_type,
                "error_message": exc.original_message,
                "suggested_repair": exc.suggested_repair,
            }
        return _quarantine_corrupt_legacy(ledger, exc)
    return {
        "status": "HEALTHY",
        "modified": False,
        "legacy_file": str(ledger.path),
        "rows": len(frame),
    }


def _quarantine_corrupt_legacy(
    ledger: TushareUsageLedger,
    error: TushareUsageLedgerCorruptError,
) -> dict[str, Any]:
    with ledger.lock_manager.resource_lock(ledger.path):
        if not ledger.path.exists():
            raise FileNotFoundError(f"legacy usage ledger no longer exists: {ledger.path}")
        digest = _sha256(ledger.path)
        size = ledger.path.stat().st_size
        quarantine_path = _timestamped_path(ledger.path.parent / "corrupt", "")
        quarantine_path.parent.mkdir(parents=True, exist_ok=True)
        sidecar_path = quarantine_path.with_suffix(quarantine_path.suffix + ".json")
        sidecar = {
            "original_path": str(error.path),
            "quarantine_path": str(quarantine_path),
            "file_size": size,
            "sha256": digest,
            "error_type": error.error_type,
            "error_message": error.original_message,
            "quarantined_at": datetime.now(tz=UTC).isoformat(),
        }
        with ledger.database_coordinator.write_transaction(
            "quarantine_tushare_usage_history_reset"
        ) as connection:
            _record_history_reset_locked(
                ledger,
                connection,
                reason="legacy_usage_ledger_corrupt",
                legacy_corrupt_file=str(quarantine_path),
            )
        try:
            _atomic_write_json(sidecar_path, sidecar)
            os.replace(ledger.path, quarantine_path)
        except Exception:
            sidecar_path.unlink(missing_ok=True)
            raise
    if ledger._data_lake is not None:
        ledger._data_lake.mark_legacy_ledger_initialized(error=None)
    ledger._ready = False
    return {
        "status": "QUARANTINED",
        "modified": True,
        "history_reset": True,
        "quarantine_path": str(quarantine_path),
        "sidecar_path": str(sidecar_path),
        "warning": (
            "Tushare usage history and request cache could not be recovered; "
            "actual remote quota usage today may be higher than local records."
        ),
    }


def _read_complete_legacy(path: Path) -> pd.DataFrame:
    try:
        parquet = pq.ParquetFile(path)  # type: ignore[no-untyped-call]
        tables = [
            parquet.read_row_group(index)  # type: ignore[no-untyped-call]
            for index in range(parquet.num_row_groups)
        ]
        if not tables:
            return pd.read_parquet(path)
        return pd.concat([table.to_pandas() for table in tables], ignore_index=True)
    except Exception as exc:
        raise TushareUsageLedgerCorruptError(path, exc) from exc


def _validate_columns(path: Path, frame: pd.DataFrame) -> None:
    missing = sorted(_REQUIRED_COLUMNS.difference(frame.columns))
    if missing:
        cause = ValueError(f"legacy ledger missing required columns: {missing}")
        raise TushareUsageLedgerCorruptError(path, cause)


def _normalize_legacy_frame(path: Path, frame: pd.DataFrame) -> pd.DataFrame:
    normalized: list[dict[str, Any]] = []
    try:
        for payload in frame.to_dict(orient="records"):
            record = _row_to_record(payload)
            if record.token_hash is not None and not _valid_sha256(record.token_hash):
                raise ValueError("token_hash must be a SHA-256 hex digest")
            record = record.model_copy(
                update={"params_redacted": redact_params(record.params_redacted)}
            )
            normalized.append(_record_to_row(record))
    except Exception as exc:
        cause = ValueError(f"legacy ledger record validation failed: {exc}")
        raise TushareUsageLedgerCorruptError(path, cause) from exc
    return pd.DataFrame(normalized, columns=sorted(_REQUIRED_COLUMNS))


def _valid_sha256(value: str) -> bool:
    return len(value) == 64 and all(character in "0123456789abcdef" for character in value.lower())


def _record_history_reset_locked(
    ledger: TushareUsageLedger,
    connection: Any,
    *,
    reason: str,
    legacy_corrupt_file: str,
) -> None:
    payload = json.dumps(
        {
            "history_reset_at": datetime.now(tz=UTC).replace(tzinfo=None).isoformat(),
            "history_reset_reason": reason,
            "legacy_corrupt_file": legacy_corrupt_file,
        },
        sort_keys=True,
    )
    connection.execute(
        f"""
        INSERT INTO {ledger.state_table_name} VALUES ('history_reset', ?, current_timestamp)
        ON CONFLICT (key) DO UPDATE
        SET value_json = excluded.value_json, updated_at = excluded.updated_at
        """,
        [payload],
    )


def _pending_migration_for_source(
    connection: Any,
    source_path: Path,
) -> tuple[Any, ...] | None:
    row = connection.execute(
        """
        SELECT migration_id, imported_rows, skipped_rows, archive_path
        FROM tushare_usage_migrations_v1
        WHERE source_path = ? AND status = 'IMPORT_COMMITTED_PENDING_ARCHIVE'
        ORDER BY migrated_at DESC
        LIMIT 1
        """,
        [str(source_path)],
    ).fetchone()
    return cast(tuple[Any, ...] | None, row)


def _finalize_migration(ledger: TushareUsageLedger, migration_id: str) -> None:
    with ledger.database_coordinator.write_transaction(
        "finalize_tushare_usage_migration"
    ) as connection:
        connection.execute(
            """
            UPDATE tushare_usage_migrations_v1
            SET status = 'MIGRATED'
            WHERE migration_id = ?
            """,
            [migration_id],
        )


def _finalize_archived_pending_migrations(ledger: TushareUsageLedger) -> None:
    if not ledger.duckdb_path.exists():
        return
    with ledger.database_coordinator.write_transaction(
        "resume_tushare_usage_migration"
    ) as connection:
        rows = connection.execute(
            """
            SELECT migration_id, source_path, archive_path
            FROM tushare_usage_migrations_v1
            WHERE status = 'IMPORT_COMMITTED_PENDING_ARCHIVE'
            """
        ).fetchall()
        for migration_id, source_path, archive_path in rows:
            if not Path(str(source_path)).exists() and Path(str(archive_path)).exists():
                connection.execute(
                    """
                    UPDATE tushare_usage_migrations_v1
                    SET status = 'MIGRATED'
                    WHERE migration_id = ?
                    """,
                    [migration_id],
                )


def _timestamped_path(directory: Path, marker: str) -> Path:
    timestamp = datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%S.%fZ")
    middle = f".{marker}" if marker else ""
    return directory / f"tushare_usage_ledger{middle}.{timestamp}.parquet"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    temp_path = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        with temp_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
    finally:
        temp_path.unlink(missing_ok=True)
