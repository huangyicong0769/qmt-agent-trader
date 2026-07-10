"""Migration and explicit recovery for the legacy Tushare usage Parquet ledger."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any
from uuid import uuid4

import pandas as pd
import pyarrow.parquet as pq

from qmt_agent_trader.data.providers.tushare.quota import (
    TushareUsageLedgerCorruptError,
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
        return {"status": "NO_LEGACY_FILE", "imported_rows": 0, "skipped_rows": 0}
    with ledger.mutation_lock():
        if not ledger.path.exists():
            return {"status": "NO_LEGACY_FILE", "imported_rows": 0, "skipped_rows": 0}
        frame = _read_complete_legacy(ledger.path)
        _validate_columns(ledger.path, frame)
        archive_path = _timestamped_path(ledger.path.parent / "archive", "migrated")
        archive_path.parent.mkdir(parents=True, exist_ok=True)
        imported = 0
        skipped = 0
        with ledger.connect() as connection:
            ledger.ensure_tables(connection)
            connection.execute("BEGIN TRANSACTION")
            try:
                before = connection.execute(
                    f"SELECT count(*) FROM {ledger.table_name}"
                ).fetchone()
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
                after = connection.execute(
                    f"SELECT count(*) FROM {ledger.table_name}"
                ).fetchone()
                if before is None or after is None:
                    raise RuntimeError("DuckDB did not return usage row counts during migration")
                imported = int(after[0]) - int(before[0])
                skipped = len(frame) - imported
                connection.execute("COMMIT")
            except Exception:
                connection.execute("ROLLBACK")
                raise
        os.replace(ledger.path, archive_path)
        with ledger.connect() as connection:
            ledger.ensure_tables(connection)
            connection.execute(
                """
                INSERT INTO tushare_usage_migrations_v1 VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    str(uuid4()),
                    datetime.now(tz=UTC).replace(tzinfo=None),
                    str(ledger.path),
                    str(archive_path),
                    imported,
                    skipped,
                    "MIGRATED",
                    None,
                ],
            )
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
    with ledger.mutation_lock():
        digest = _sha256(ledger.path)
        size = ledger.path.stat().st_size
        quarantine_path = _timestamped_path(ledger.path.parent / "corrupt", "")
        quarantine_path.parent.mkdir(parents=True, exist_ok=True)
        os.replace(ledger.path, quarantine_path)
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
        _atomic_write_json(sidecar_path, sidecar)
    ledger.record_history_reset(
        reason="legacy_usage_ledger_corrupt",
        legacy_corrupt_file=str(quarantine_path),
    )
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
