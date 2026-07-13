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

## Destructive preserve-raw reset

Use this only when an incompatible storage refactor makes existing control state and Agent
artifacts disposable. The profile preserves validated provider data under `data/lake/raw` and
removes derived lake layers, control metadata, current and legacy registries, sessions, reports, approvals, order
plans, generated code, audit logs, backups, and quarantine evidence.

First create a read-only plan:

```bash
uv run qmt-agent storage reset --profile preserve-raw --dry-run
```

Review `delete_paths`, `file_count`, `byte_count`, and the preserved raw totals. Execute only with
the exact digest returned by that plan:

```bash
uv run qmt-agent storage reset --profile preserve-raw --confirm <digest>
```

The digest is bound to file paths, sizes, and hashes. Any intervening change rejects execution.
The command refuses corrupt raw Parquet, non-Parquet raw files, or symbolic links, excludes
cooperating writers through the maintenance barrier, and stages removals on the same filesystem.
Failures before schema verification and receipt creation restore the targets that were moved into
staging. Targets that were never moved remain untouched; `rollback_failed` requires manual
inspection of the reported staging directory.

After verification and receipt creation, the reset is considered complete. If removal of the
staging directory fails, the command keeps the verified new state and returns `status: completed`
together with `reason` and `staging_path`. Inspect the reported directory and remove it manually
after confirming the reset result.

Successful resets write a content-free receipt under `data/storage-resets/`. This workflow is not
a migration or backup facility: successful execution permanently deletes the old state.

`verify` is strictly read-only. The default checks parsable metadata and durable
envelopes; `--deep` reads every Parquet row group. A non-healthy result exits 1.
For order-plan governance, a valid event stream is insufficient by itself:
verification also requires the bound order-plan manifest and content to pass schema-v1,
byte-length, content-hash, identity, and relative-path checks. Orphan streams and streams
bound to missing or tampered plans are unhealthy.

Event-stream diagnostics distinguish the operator action required:

- `ORPHAN_EVENT_STREAM`: the stream has no valid event identity, including an empty
  stream. Append is refused; `storage quarantine order_plan_events <file>` is allowed.
- `INVALID_ORDER_PLAN`: the stream points to an artifact whose envelope,
  `OrderPlan` payload, identity, or `plan_hash` is invalid.
- `MISSING_ORDER_PLAN`: the stream points to a missing manifest or content file.

Once verification detects a valid event identity, plan-binding failures are classified
exclusively as `MISSING_ORDER_PLAN` or `INVALID_ORDER_PLAN`; they do not overlap with
`ORPHAN_EVENT_STREAM`.

For a failed first JSONL write, rollback restores the stream's prior absence.
Directory-fsync failure after a successful append is different: durability is
uncertain, the event may already be present, and the operation does not claim
that the append was rolled back.

Run `migrate --dry-run` first: it reads the migration registry without creating
the database or directories. The apply command runs only the immutable built-in
non-destructive catalog; destructive migrations require a separate explicitly
approved API call and are not exposed by this CLI.

## Backup and recovery

Backup v1 is local only. The service takes the global backup barrier, checkpoints
DuckDB through `DatabaseCoordinator`, copies every cataloged official store,
hashes every copy, and verifies staging before publishing it.
Cache, temporary files, active lock files, and prior backups are excluded. The
configured `reports/cache` and transient `reports/tool_payloads` transport
are also excluded; governed backtest/research reports and agent-generated code
remain included through their exact composition roots. A
`SUCCESS.json` marker bound to the manifest hash is written only after exact-set,
contained-path, size, and hash verification. Copy, hash, or snapshot-validation
failures raise `StorageBackupError`; lock contention remains a structured
`StorageLockTimeoutError` or `StorageConflictError`. All failures remove staging
and incomplete final data.

All canonical filesystem and database writers enter the same mutation gate at
`PersistencePaths.locks_root`; this includes governed artifacts, audit streams,
generated code, and incremental and nonincremental Parquet publication. Backup
holds the exclusive gate across enumeration and copy. Writers therefore wait for
the local backup duration; long backups trade write latency for a single
consistent generation. Readers remain available subject to DuckDB checkpoint
coordination.

After byte/hash verification, backup re-runs deep storage health against rebased
snapshot paths. Copied DuckDB migrations/schema, Parquet pages, structured
documents, JSONL streams, and governed manifest bindings must all be healthy
before the success marker can be published.

Recovery is deliberately manual in v1:

1. Stop writers and inspect `storage locks`; do not delete a live lock.
2. Verify the selected backup manifest and every listed SHA-256 digest.
3. Restore into a separate staging project root.
4. Run `storage verify --deep` against staging.
5. Atomically select the staged root using the deployment's local procedure.

Quarantine is never automatic for authoritative data. The command accepts only
an exact catalog store and contained record path and rejects healthy records.
Ordinary records use their canonical resource lock. Governed artifacts use the
artifact-root lock and move content plus manifest as one rollback-safe unit;
an order plan also moves its corresponding governance event stream. The
quarantine operation records whether content was present, already missing, or
unknown because an invalid manifest had no recoverable binding. A parseable invalid
manifest may bind content only when its artifact id, safe relative path, and hashed
manifest filename agree; otherwise only the manifest is isolated and unrelated
orphans remain untouched. The sidecar records binding state, hashes, paths,
diagnostics, and time. Evidence
publication failure rolls the complete unit back. Tushare's existing repair
command remains the owner of its specialized ledger history-reset protocol.

Artifact manifests support schema v1 only. Their `byte_length` and `content_hash`
are independent strong invariants, and a manifest continues to own its contained
relative path even if content is missing. That ownership ends only when the
manifest is removed or isolated by quarantine; missing content alone never frees
the path for a different artifact.

## Lock order and limitations

The backup barrier precedes resource locks, which precede the DuckDB write lock.
`storage locks` maps known hashes through the same catalog/LockManager contract;
unknown hashes are labeled unknown. Active probing never breaks a live lock and
mtime-only stale status is diagnostic, not permission to delete.

The backup destination shares the local data root. It protects against logical
or operator damage when retained, but not disk loss, host loss, or distributed
writers. Retention, encryption, off-device copies, restore selection, and cloud
support require a later evidence-backed design.

Catalog layout is declarative: each `StoreDefinition` keeps its configured
single-file or directory layout regardless of whether that path currently exists.
