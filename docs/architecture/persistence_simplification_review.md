# Persistence Simplification Review

Baseline: `e349317cc0c3fb13f0caac02fa4f4d00aafa8a04` (`main`). The baseline passed
643 tests, Ruff, and mypy. Persistence production modules contained 3,745 lines.

## Lock model before this work

Every resource writer and DuckDB writer acquired the same exclusive
`backup-barrier.lock` before its own lock. DuckDB access also used an in-process
reader/writer gate, a cross-process writer lock, and DuckDB locking. As a result,
unrelated file writers were globally serialized and long readers could prevent
control-plane writes.

## Runtime paths under review

- File repositories: resource lock, read/modify/write, atomic replace.
- DuckDB control state: in-process gate, database file lock, transaction.
- Backup: global barrier, checkpoint, copy, manifest, verification, success marker.
- Verify/quarantine: catalog traversal plus generic extension-based validation.
- Governed artifacts: manifest/hash verification and immutable content files.

## Mechanisms removed or narrowed

- The long-held exclusive backup lock on every ordinary writer.
- The in-process DuckDB reader/writer gate.
- Read-time schema migration and artifact adoption.
- Generic JSON validity checks that disagree with repository validation.
- Broad exception swallowing in cache and append-only readers.
- Compatibility paths that re-expose raw Parquet exceptions.

## Reliability boundaries retained

- Per-resource mutual exclusion and atomic file replacement.
- One canonical cross-process DuckDB writer lock and transactions.
- Backup admission control that prevents new writers and drains active writers.
- Current schema-v2 validation, content hashes, identity checks, and manifests.
- Structured storage errors and fail-closed trading governance.

## Intended behavior changes

Only current schema-v2 state and manifested governed artifacts are accepted.
Reads never migrate, adopt, create, or rewrite persistence. Storage diagnostics
use the same validators as runtime reads. Independent writers may proceed in
parallel; backup remains a deliberate maintenance window.
