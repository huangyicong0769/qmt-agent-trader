# ADR: retain DuckDB and introduce a coordinated control-store boundary

- Status: accepted for the persistence refactor
- Date: 2026-07-10
- Scope: local analytical catalog and control-plane records

## Context

The application currently places the analytical catalog, generic fetch state/events, and the completed Tushare usage ledger in `data/qmt_agent_trader.duckdb`. Other mutable control state is spread across whole-file JSON documents. DuckDB access has two owners: `DataLake.connect` has no application coordinator, while `TushareUsageLedger.connect` protects ledger mutations with a ledger-specific file lock. This split cannot establish one ordering rule for all writes to the database file.

The refactor needs interfaces and safer coordination without changing persistence technology based on an unmeasured assumption. In particular, the Tushare ledger already provides request-id idempotency, transactions, legacy recovery, quarantine, structured errors, and audit tables. Replatforming it now would combine a concurrency change with a schema/data migration and weaken rollback.

## Options considered

### 1. One DuckDB file with a global database write coordinator/lock

Retain `data/qmt_agent_trader.duckdb`. All DDL/DML is routed through one injected coordinator that owns the inter-process file lock, connection lifecycle, transaction boundary, timeout/error mapping, and write observability. Reads use an explicit coordinator policy rather than opening ad hoc connections during a write.

- Benefits: smallest behavioral change; preserves current tables and legacy recovery; one write-ordering rule; reversible interface extraction; analytical views and control records remain locally queryable together.
- Costs: serializes writes; DuckDB remains a mixed analytical/control file; long transactions can delay latency-sensitive state; correctness depends on every writer using the coordinator.

### 2. Analytical DuckDB plus SQLite WAL control store

Move mutable control records to SQLite in WAL mode while retaining DuckDB for Parquet views and analytical metadata.

- Benefits: SQLite is designed for transactional local application state and WAL supports readers alongside a writer; separates control schema lifecycle from the catalog.
- Costs: an immediate data migration, dual backup/restore consistency, new operational failure modes, and cross-store operations without atomic commit. It would require translating and validating the completed Tushare ledger and its recovery audit before there is reproducible evidence that DuckDB plus coordination is insufficient.

### 3. Local control service

Put control operations behind a single local daemon/service with an RPC/API boundary and let it own its database.

- Benefits: naturally centralizes serialization, policy, migrations, and observability; allows later storage replacement behind the service.
- Costs: process supervision, authentication, availability, protocol versioning, deployment, and recovery complexity are disproportionate for the current local application. It adds a new runtime dependency and failure domain during a persistence-only refactor.

## Decision

Use option 1 for this refactor: retain the current DuckDB file, introduce control-store and analytical-catalog interfaces, and route **every** DuckDB mutation through one global write coordinator and lock. The coordinator must be injected from the composition root; neither `DataLake` nor Tushare code may create an independent mutation lock after migration.

This decision does not declare DuckDB permanently optimal for control state. A SQLite migration requires a separate ADR supported by a reproducible multi-process stress test that demonstrates unmet correctness or latency requirements under the coordinated DuckDB design. Anecdotal lock errors, synthetic single-process benchmarks, or preference alone are not sufficient.

## Required coordinator contract

- One canonical database path and one canonical lock path derived from injected configuration.
- Bounded lock acquisition with a structured timeout error and operation identity.
- Explicit transactions for multi-statement state changes, including current-state plus event append.
- Idempotency keys or compare-and-set semantics where duplicate/retried control operations are possible.
- No direct production `duckdb.connect` outside the coordinator after migration.
- Observable wait duration, transaction duration, owner/process identity, outcome, and rollback.
- Re-entrant behavior must be explicit; nested operations may not deadlock or silently open a second writer.
- Read policy must define whether readers wait, use a read-only connection, or read a stable snapshot while a writer is active.

## Tushare ledger preservation

The existing tables remain `tushare_usage_events_v1`, `tushare_usage_state_v1`, and `tushare_usage_migrations_v1`. The refactor preserves request-id conflict handling, redacted parameters, transaction rollback, legacy Parquet import, pending-archive resume, corruption quarantine and SHA-256 sidecar, history-reset warnings, structured `TushareUsageLedgerCorruptError`, and the repair command contract. Its `.tushare.lock` behavior is folded into—not run alongside—the global coordinator only after equivalent tests exist.

## Consequences and risks

- Database writes become serial and may queue; the coordinator therefore needs bounded waits and metrics.
- A global lock can amplify a slow writer into broad control-plane latency. Transactions must avoid network calls and large DataFrame work inside the critical section.
- Mixed analytical/control backup remains a limitation. A DuckDB file backup must be coordinated with Parquet/catalog generation metadata; copying a live file ad hoc is not a recovery plan.
- Interface extraction creates an enforcement problem: CI/static checks should prevent new direct connections and DDL/DML outside approved owners.
- File locks depend on filesystem semantics. The supported deployment remains local storage unless a later ADR proves a different lock/service design.
- The coordinator cannot make DuckDB and Parquet replacement one atomic transaction. Catalog reconciliation/rebuild and manifests remain necessary.

## Verification gate and exit criteria

Before claiming the coordinated design sufficient, tests must spawn real OS processes against the same database and exercise, at minimum:

1. concurrent Tushare event appends with duplicate request ids;
2. generic fetch current-state replacement plus event append;
3. catalog view registration while control writes are queued;
4. forced process termination before commit, during transaction, and after commit;
5. lock timeout and recovery after owner exit;
6. startup recovery of `IMPORT_COMMITTED_PENDING_ARCHIVE` legacy migrations;
7. validation that no committed row is lost, duplicated contrary to its idempotency contract, or separated from its required event.

The harness must record process count, operation mix, random seed, platform/filesystem, timings, failures, final invariants, and be reproducible in CI or a documented local command.

A separate SQLite ADR may be proposed if this evidence repeatedly shows an invariant violation that cannot be fixed without defeating the architecture, or if measured p95/p99 wait time exceeds a separately approved service-level target under representative workload. A local control service requires evidence that a database split alone is insufficient or that multiple independently deployed clients need a stable network boundary.

## Rollback

Interface and coordinator rollout must be incremental. The physical DuckDB schema is unchanged, so rollback means reverting callers to the prior repository implementations while retaining the same database and Tushare records. Do not delete or rewrite ledger tables as part of rollback. If the coordinator causes operational regression, disable only the new composition path after stopping writers; verify DuckDB integrity and pending migration state before resuming the old path.

## Non-goals

This ADR does not change factor/backtest logic, QMT gateway behavior, UI, trading automation, approvals, `dry_run`, or live-trading flags. It does not authorize a SQLite migration.
