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

Backup v1 is local only. The service takes the global backup barrier, checkpoints
DuckDB through `DatabaseCoordinator`, copies every cataloged official store,
hashes every copy, and verifies staging before publishing it.
Cache, temporary files, active lock files, and prior backups are excluded. A
`SUCCESS.json` marker bound to the manifest hash is written only after exact-set,
contained-path, size, and hash verification. A failure raises
`StorageBackupError` and removes staging and incomplete final data.

Recovery is deliberately manual in v1:

1. Stop writers and inspect `storage locks`; do not delete a live lock.
2. Verify the selected backup manifest and every listed SHA-256 digest.
3. Restore into a separate staging project root.
4. Run `storage verify --deep` against staging.
5. Atomically select the staged root using the deployment's local procedure.

Quarantine is never automatic for authoritative data. The command accepts only
an exact catalog store and contained record path, takes the record lock before
type-aware deep validation, rejects healthy Parquet/JSON/JSONL/YAML/code and
governed artifacts, moves the requested bytes, and writes hash/size/path/time
evidence. Evidence failure rolls the original bytes back. Tushare's existing repair command remains
the owner of its specialized ledger history-reset protocol.

## Lock order and limitations

The backup barrier precedes resource locks, which precede the DuckDB write lock.
`storage locks` maps known hashes through the same catalog/LockManager contract;
unknown hashes are labeled unknown. Active probing never breaks a live lock and
mtime-only stale status is diagnostic, not permission to delete.

The backup destination shares the local data root. It protects against logical
or operator damage when retained, but not disk loss, host loss, or distributed
writers. Retention, encryption, off-device copies, restore selection, and cloud
support require a later evidence-backed design.
