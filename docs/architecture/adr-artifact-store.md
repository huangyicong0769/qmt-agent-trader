# ADR: files remain the artifact store behind shared safe-write APIs

- Status: accepted for the persistence refactor
- Date: 2026-07-10
- Scope: JSON, YAML, Markdown, Python, JSONL, and related manifests

## Context

Reports, approvals, order plans, generated code, audit logs, proposals, and several mutable registries are files because people need to inspect, diff, archive, and review them without a database client. That property is valuable. The problem is not the file format: production writers currently use independent `write_text` or append calls, usually without atomic replacement, inter-process locking, create-only enforcement, hashes, manifests, recovery, or injected roots.

Governance artifacts are especially sensitive. Strategy approval YAML and approved order-plan JSON must not be silently replaced by a later write using the same identifier. Generated reports/code must retain the exact bytes reviewed. Mutable documents such as registries and chat sessions need revisions rather than pretending to be immutable.

## Decision

Keep JSON, YAML, Markdown, Python, and JSONL artifacts as files. Route every production write through an injected artifact-store interface with three explicit operations:

1. **create-only immutable artifact:** fail if the logical/physical identity already exists;
2. **revisioned mutable document:** create an immutable revision and atomically advance a current pointer using an expected revision;
3. **locked append stream:** append framed records under an inter-process lock with tail recovery rules.

No production module may call `Path.write_text`, `write_bytes`, append-mode `open`, or an unrestricted filesystem move for these domains after migration. Reads and deletes also go through domain repositories so root policy and recovery are consistent.

## Immutable artifacts

Approval YAML, order plans, research/backtest run reports, generated code candidates, tool proposals, quarantine evidence, and immutable audit segments use create-only semantics. The store writes to a same-filesystem temporary path, flushes and `fsync`s the file, validates the serialized form where applicable, atomically publishes without replacing an existing target, and `fsync`s the directory.

Strategy approval and order-plan governance is preserved:

- `{strategy_id, strategy_version}` approval identity is immutable. A correction, revocation, or superseding decision is a new artifact linked to the prior decision; it never edits approved bytes in place.
- `order_plan_id` is immutable. Saving an existing id is a conflict even when content appears equal; an idempotent retry may succeed only after verifying the stored content hash matches exactly.
- Storage APIs cannot upgrade `approval_status`, enable live trading, or bypass existing model/workflow checks. They only persist authorized domain objects.
- Research and generated artifacts remain explicitly non-live/review-required where their models currently say so.

## Revisioned mutable documents

Factor, strategy, and universe registries; experiment and todo records; chat sessions; and mutable Markdown syntheses receive a stable logical id plus monotonically increasing revision. Each revision is immutable. An atomic current pointer contains the selected revision and content hash. Writers supply `expected_revision`; a mismatch is a structured conflict rather than last-writer-wins.

Deleting a mutable document creates a tombstone revision. Retention/compaction may remove old revisions only under an explicit policy and never when they are referenced by an immutable artifact or audit manifest. Disposable caches may use replaceable atomic entries and expiry instead of durable revision history, but still require injected roots and coordination.

## Manifests and hashes

Every immutable artifact/revision has a sidecar or collection manifest containing at least:

- storage schema version, artifact type, logical id, revision (when applicable), and created timestamp;
- SHA-256 of the exact content bytes and byte length;
- producer/module and relevant run/session/experiment ids;
- content media type and domain schema/model version when one exists;
- links to predecessor/superseded artifacts and referenced evidence;
- governance flags needed for review, without inventing approval.

Manifests are published atomically with deterministic naming. Where the filesystem cannot atomically commit content and manifest together, the store uses a pending marker and deterministic startup reconciliation: complete a valid pair or move incomplete material to quarantine. Readers verify hashes before returning governance-critical artifacts.

## JSONL append streams

Audit JSONL remains human-readable, but append goes through a single locked API. Each record is serialized to one bounded line, assigned stream/version and sequence metadata, and written under an inter-process lock. The writer flushes and `fsync`s before acknowledging. Readers tolerate only a documented incomplete final record: they report and quarantine/truncate it through an explicit repair operation, never silently skip malformed interior records. Rotation produces immutable, hashed segments with a manifest linking segment order.

## Quarantine and recovery

Deserialization, schema validation, or hash failure does not trigger silent deletion for durable state. The store moves suspect bytes to a configured quarantine root using collision-resistant names and writes evidence containing original path, reason, size, hash when readable, producer/schema hints, and quarantine time. Governance artifacts are never auto-repaired by overwriting them. Cache entries may be dropped under their explicit disposable policy.

Backups operate on immutable generations/segments plus manifests. A backup is successful only when its manifest verifies; restoration goes to a staging root, validates hashes/schema, then atomically selects a generation. Retention and off-device destination are deployment policy and must be defined before persistence is described as recoverable. Existing Tushare legacy archive/quarantine behavior remains intact and is adapted, not discarded.

## Path injection and containment

The composition root injects typed roots for artifacts, control documents, caches, sessions, audit streams, generated code, and quarantine. Production code does not derive roots from `Path.cwd()`, module location, or unvalidated request payloads.

- Logical ids are encoded by one path policy; replacing `/` alone is insufficient.
- The resolved target must remain beneath its configured root, with symlink/escape handling defined.
- Callers may select a logical namespace, never an arbitrary absolute path.
- Generated code retains `CodeSandbox` containment and static scanning in addition to artifact-store checks.
- Root changes are explicit configuration changes and are recorded in manifests/diagnostics without leaking secrets.

## Consequences and risks

- Human-readable files and normal review/diff workflows remain available.
- More files and manifests are created; retention, indexing, and cleanup must be implemented deliberately.
- Create-only publication and atomic pointer updates vary by platform. The implementation must test the actual macOS/local-filesystem primitives and fail closed when exclusivity is uncertain.
- Inter-process locks can queue writers. Critical sections must contain only serialization/publication work, with structured timeouts and metrics.
- Hashes detect corruption/tampering but do not prove authorship. Signing approvals is a separate security decision.
- A manifest protocol adds partial-commit states; pending markers and startup reconciliation are mandatory, not optional polish.

## Rollout and rollback

Introduce the interface and adapters domain by domain. Dual-write is not the default because it creates two competing sources of truth; use offline import or a bounded, verified migration per domain. A migrated reader may temporarily support read-old/write-new with a recorded import marker, but must not overwrite legacy governance artifacts.

Rollback selects the last verified artifact generation/current pointer and reverts the calling adapter. Immutable originals, approval YAML, order plans, audit segments, and manifests are retained. Quarantined material is never automatically promoted during rollback.

## Acceptance criteria

- Repository searches show no unapproved production direct text/byte/append writes for inventoried domains.
- Multi-process tests prove create-only conflict behavior, expected-revision conflict behavior, and serialized append behavior.
- Kill-point tests cover temp write, file `fsync`, publish, directory `fsync`, manifest/pointer publication, and startup reconciliation.
- Approval and order-plan tests prove same-id replacement fails and content-hash-identical retry is explicitly handled.
- Path tests cover absolute paths, `..`, separators, Unicode/case collisions, symlinks, and injected non-default roots.
- Corruption tests prove durable state is reported/quarantined rather than silently deleted, while caches follow their documented disposable policy.

## Non-goals

This ADR does not move artifacts into SQLite/DuckDB, alter strategy/factor/backtest behavior, change QMT/UI/trading automation, change approval transitions, or enable live trading. It does not weaken the existing atomic Parquet writer or Tushare recovery path.
