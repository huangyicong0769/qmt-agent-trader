# Local storage operations

Phase 6 provides one operator surface for the existing file, Parquet, artifact,
JSONL, and DuckDB stores. It does not provide cloud, distributed, or off-device
backup.

## Commands

```bash
uv run qmt-agent storage inventory
uv run qmt-agent storage verify
uv run qmt-agent storage verify --deep
uv run qmt-agent storage migrate --dry-run
uv run qmt-agent storage migrate
uv run qmt-agent storage backup
uv run qmt-agent storage locks
uv run qmt-agent storage quarantine sessions bad-record.json
```

`verify` is strictly read-only. The default checks parsable metadata and durable
envelopes; `--deep` reads every Parquet row group. A non-healthy result exits 1.
Run `migrate --dry-run` first: it reads the migration registry without creating
the database or directories. The apply command runs only the immutable built-in
non-destructive catalog; destructive migrations require a separate explicitly
approved API call and are not exposed by this CLI.

## Backup and recovery

Backup v1 is local only. The service takes resource barriers in sorted canonical
path order, copies official mutable data and governed artifacts, hashes every
copy, writes a manifest, and verifies the staging backup before publishing it.
Cache, temporary files, active lock files, and prior backups are excluded. A
`SUCCESS.json` marker is written only after verification. A failure raises
`StorageBackupError` and removes staging data.

Recovery is deliberately manual in v1:

1. Stop writers and inspect `storage locks`; do not delete a live lock.
2. Verify the selected backup manifest and every listed SHA-256 digest.
3. Restore into a separate staging project root.
4. Run `storage verify --deep` against staging.
5. Atomically select the staged root using the deployment's local procedure.

Quarantine is never automatic for authoritative data. The command accepts only
an allowlisted canonical store and contained record path, takes the record lock,
rejects valid JSON/YAML authoritative records, moves the requested bytes, and
writes hash/size/path/time evidence. Tushare's existing repair command remains
the owner of its specialized ledger history-reset protocol.

## Lock order and limitations

Resource locks precede the DuckDB write lock. Backup barriers acquire canonical
resources in sorted absolute-path order. Operators must never break a lock that
reports active; stale status is diagnostic, not permission to delete.

The backup destination shares the local data root. It protects against logical
or operator damage when retained, but not disk loss, host loss, or distributed
writers. Retention, encryption, off-device copies, restore selection, and cloud
support require a later evidence-backed design.
