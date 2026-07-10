"""Explicit startup initialization for the local persistence system."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from qmt_agent_trader.persistence.migrations import Migration, MigrationRegistry

if TYPE_CHECKING:
    from qmt_agent_trader.data.storage import DataLake


FETCH_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS data_fetch_state_v2 (
    source TEXT NOT NULL,
    dataset_id TEXT NOT NULL,
    api_name TEXT NOT NULL,
    endpoint_id TEXT NOT NULL,
    param_hash TEXT NOT NULL,
    fields_hash TEXT NOT NULL,
    symbols_hash TEXT NOT NULL,
    fetched_at TIMESTAMP NOT NULL,
    coverage_start TEXT,
    coverage_end TEXT,
    row_count BIGINT NOT NULL,
    checksum TEXT,
    status TEXT NOT NULL,
    error TEXT,
    PRIMARY KEY (
        source, dataset_id, api_name, endpoint_id,
        param_hash, fields_hash, symbols_hash
    )
);
CREATE TABLE IF NOT EXISTS data_fetch_events_v2 (
    source TEXT NOT NULL,
    dataset_id TEXT NOT NULL,
    api_name TEXT NOT NULL,
    endpoint_id TEXT NOT NULL,
    param_hash TEXT NOT NULL,
    fields_hash TEXT NOT NULL,
    symbols_hash TEXT NOT NULL,
    fetched_at TIMESTAMP NOT NULL,
    coverage_start TEXT,
    coverage_end TEXT,
    row_count BIGINT NOT NULL,
    checksum TEXT,
    status TEXT NOT NULL,
    error TEXT
);
CREATE TABLE IF NOT EXISTS data_fetch_state (
    source TEXT NOT NULL,
    dataset TEXT NOT NULL,
    start_date TEXT NOT NULL,
    end_date TEXT NOT NULL,
    status TEXT NOT NULL,
    row_count BIGINT NOT NULL,
    checksum TEXT,
    updated_at TIMESTAMP NOT NULL,
    error TEXT,
    PRIMARY KEY (source, dataset, start_date, end_date)
);
CREATE TABLE IF NOT EXISTS data_fetch_events (
    source TEXT NOT NULL,
    dataset TEXT NOT NULL,
    start_date TEXT NOT NULL,
    end_date TEXT NOT NULL,
    status TEXT NOT NULL,
    row_count BIGINT NOT NULL,
    checksum TEXT,
    updated_at TIMESTAMP NOT NULL,
    error TEXT
)
""".strip()

TUSHARE_USAGE_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS tushare_usage_events_v1 (
    request_id TEXT PRIMARY KEY,
    run_id TEXT,
    api_name TEXT NOT NULL,
    params_hash TEXT NOT NULL,
    params_redacted TEXT NOT NULL,
    fields TEXT NOT NULL,
    planned_at TIMESTAMP,
    started_at TIMESTAMP,
    finished_at TIMESTAMP,
    status TEXT NOT NULL,
    row_count BIGINT,
    error_type TEXT,
    error_message TEXT,
    token_hash TEXT,
    execution_mode TEXT NOT NULL,
    created_at TIMESTAMP NOT NULL
);
CREATE TABLE IF NOT EXISTS tushare_usage_state_v1 (
    key TEXT PRIMARY KEY,
    value_json TEXT NOT NULL,
    updated_at TIMESTAMP NOT NULL
);
CREATE TABLE IF NOT EXISTS tushare_usage_migrations_v1 (
    migration_id TEXT PRIMARY KEY,
    migrated_at TIMESTAMP NOT NULL,
    source_path TEXT NOT NULL,
    archive_path TEXT NOT NULL,
    imported_rows BIGINT NOT NULL,
    skipped_rows BIGINT NOT NULL,
    status TEXT NOT NULL,
    error_message TEXT
)
""".strip()

FETCH_STATE_PRIMARY_KEY_UPGRADE_SQL = """
CREATE TABLE data_fetch_state_v2_pk_upgrade (
    source TEXT NOT NULL,
    dataset_id TEXT NOT NULL,
    api_name TEXT NOT NULL,
    endpoint_id TEXT NOT NULL,
    param_hash TEXT NOT NULL,
    fields_hash TEXT NOT NULL,
    symbols_hash TEXT NOT NULL,
    fetched_at TIMESTAMP NOT NULL,
    coverage_start TEXT,
    coverage_end TEXT,
    row_count BIGINT NOT NULL,
    checksum TEXT,
    status TEXT NOT NULL,
    error TEXT,
    PRIMARY KEY (
        source, dataset_id, api_name, endpoint_id,
        param_hash, fields_hash, symbols_hash
    )
);
INSERT INTO data_fetch_state_v2_pk_upgrade
SELECT * EXCLUDE (dedupe_rank) FROM (
    SELECT *, row_number() OVER (
        PARTITION BY source, dataset_id, api_name, endpoint_id,
                     param_hash, fields_hash, symbols_hash
        ORDER BY fetched_at DESC, status DESC, checksum DESC NULLS LAST,
                 row_count DESC
    ) AS dedupe_rank
    FROM data_fetch_state_v2
) WHERE dedupe_rank = 1;
DROP TABLE data_fetch_state_v2;
ALTER TABLE data_fetch_state_v2_pk_upgrade RENAME TO data_fetch_state_v2;
CREATE TABLE data_fetch_state_pk_upgrade (
    source TEXT NOT NULL,
    dataset TEXT NOT NULL,
    start_date TEXT NOT NULL,
    end_date TEXT NOT NULL,
    status TEXT NOT NULL,
    row_count BIGINT NOT NULL,
    checksum TEXT,
    updated_at TIMESTAMP NOT NULL,
    error TEXT,
    PRIMARY KEY (source, dataset, start_date, end_date)
);
INSERT INTO data_fetch_state_pk_upgrade
SELECT * EXCLUDE (dedupe_rank) FROM (
    SELECT *, row_number() OVER (
        PARTITION BY source, dataset, start_date, end_date
        ORDER BY updated_at DESC, status DESC, checksum DESC NULLS LAST,
                 row_count DESC
    ) AS dedupe_rank
    FROM data_fetch_state
) WHERE dedupe_rank = 1;
DROP TABLE data_fetch_state;
ALTER TABLE data_fetch_state_pk_upgrade RENAME TO data_fetch_state
""".strip()


def _execute_script(connection: Any, script: str) -> None:
    for statement in script.split(";"):
        if statement.strip():
            connection.execute(statement)


def storage_migrations() -> tuple[Migration, ...]:
    """Return the immutable, ordered schema migration catalog."""
    return (
        Migration(
            migration_id="data-fetch-metadata-v1",
            component="data_lake",
            version=1,
            description="Create current and supported legacy fetch metadata tables",
            apply=lambda connection: _execute_script(connection, FETCH_SCHEMA_SQL),
            implementation=FETCH_SCHEMA_SQL,
        ),
        Migration(
            migration_id="data-fetch-state-primary-keys-v2",
            component="data_lake",
            version=2,
            description="Deduplicate fetch state and add stable primary keys",
            apply=lambda connection: _execute_script(
                connection, FETCH_STATE_PRIMARY_KEY_UPGRADE_SQL
            ),
            implementation=FETCH_STATE_PRIMARY_KEY_UPGRADE_SQL,
        ),
        Migration(
            migration_id="tushare-usage-store-v1",
            component="tushare_usage",
            version=1,
            description="Create Tushare usage, state, and migration audit tables",
            apply=lambda connection: _execute_script(connection, TUSHARE_USAGE_SCHEMA_SQL),
            implementation=TUSHARE_USAGE_SCHEMA_SQL,
        ),
    )


def initialize_persistence(
    lake: DataLake,
    *,
    migrate_legacy_ledger: bool = True,
    raise_on_legacy_error: bool = True,
) -> dict[str, Any]:
    """Apply pending schemas and run legacy recovery once for this lake instance."""
    if lake.persistence_schema_initialized and (
        not migrate_legacy_ledger or lake.legacy_ledger_initialized
    ):
        error = lake.persistence_initialization_error
        if error is not None and migrate_legacy_ledger and raise_on_legacy_error:
            raise error
        return {"status": "ALREADY_INITIALIZED", "legacy_error": error}

    applied = (
        []
        if lake.persistence_schema_initialized
        else MigrationRegistry(lake.database_coordinator).apply(storage_migrations())
    )
    lake.mark_persistence_schema_initialized()
    legacy_result: dict[str, Any] | None = None
    legacy_error: Exception | None = None
    legacy_attempt_completed = False
    if migrate_legacy_ledger and not lake.legacy_ledger_initialized:
        from qmt_agent_trader.data.providers.tushare.ledger_migration import (
            migrate_legacy_usage_ledger,
        )
        from qmt_agent_trader.data.providers.tushare.quota import TushareUsageLedger

        ledger = TushareUsageLedger.from_data_lake(lake)
        try:
            legacy_result = migrate_legacy_usage_ledger(ledger)
            legacy_attempt_completed = True
        except Exception as exc:
            legacy_error = exc
            from qmt_agent_trader.data.providers.tushare.quota import (
                TushareUsageLedgerCorruptError,
            )

            legacy_attempt_completed = isinstance(exc, TushareUsageLedgerCorruptError)
        if legacy_attempt_completed:
            lake.mark_legacy_ledger_initialized(error=legacy_error)
    if legacy_error is not None and raise_on_legacy_error:
        raise legacy_error
    return {
        "status": "INITIALIZED",
        "applied_migrations": applied,
        "legacy_result": legacy_result,
        "legacy_error": legacy_error,
    }
