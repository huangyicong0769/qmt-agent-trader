# Tushare usage ledger recovery

Tushare request accounting is stored transactionally in the project DuckDB database:

```text
data/qmt_agent_trader.duckdb
└── tushare_usage_events_v1
```

The former mutable ledger at
`data/lake/metadata/tushare_usage_ledger.parquet` is now a legacy input only. A healthy
legacy file is fully read, validated, imported by `request_id`, and moved to
`data/lake/metadata/archive/`. Normal operation never rewrites that Parquet file.

## Diagnose and recover a corrupt legacy ledger

The default command is read-only:

```bash
uv run qmt-agent data repair-tushare-ledger
```

If it reports `CORRUPT`, stop processes that may run Tushare fetches, then explicitly
quarantine the file:

```bash
uv run qmt-agent data repair-tushare-ledger --quarantine-corrupt
```

The command moves the file to `data/lake/metadata/corrupt/` and writes a JSON sidecar
containing the original path, quarantine path, size, SHA-256 digest, error summary, and UTC
timestamp. It does not contact Tushare.

Quarantine resets only the local usage history and equivalent-request cache. It does not reset
the real Tushare account quota. Until the next UTC day, planner output includes
`TUSHARE_USAGE_HISTORY_RESET`; treat the displayed remaining quota as a lower-confidence upper
bound and keep normal approval requirements in place.

Do not manually delete the active DuckDB database, a legacy ledger under investigation, or a
`.lock` file belonging to a running process. Preserve quarantined files and sidecars for audit.

## Failure classification

When a legacy ledger is unreadable, `plan_tushare_fetch` and `run_tushare_fetch` return
`TUSHARE_USAGE_LEDGER_CORRUPT`, `source=local_metadata`, and
`remote_request_attempted=false`. The tools do not label this as a remote Tushare API failure.

All DataLake Parquet writes use a same-directory temporary file, full row-group validation,
`fsync`, and atomic replacement. Incremental read-modify-write operations also hold a
cross-process `<dataset>.parquet.lock` for the complete merge and replacement sequence.
