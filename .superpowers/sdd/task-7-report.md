# Task 7 report — Storage health and operations

## Delivered

- A single injected `StorageOperations` service for canonical inventory,
  read-only/default and deep verification, built-in migration planning/apply,
  consistent local backup and hash verification, lock diagnostics, explicit
  path-safe quarantine, and secret-safe structured health payloads.
- `qmt-agent storage inventory|verify|migrate|backup|locks|quarantine` with JSON
  output and health/error exit codes. Existing data validation and Tushare repair
  commands remain unchanged.
- AST enforcement for the four forbidden persistence primitives. The only
  allowlisted owners are the atomic Parquet writer and database coordinator;
  remaining core direct text writes were migrated.
- Operator documentation for recovery, lock order, backup scope/limitations,
  quarantine, and evidence triggers for a future SQLite/control-service ADR.

## TDD evidence

The first focused run failed collection with two expected missing-module errors.
CLI tests then failed with three unknown-command exits. Intermediate GREEN runs
caught cache inclusion in backup, two real architecture violations, and invalid
Rich-wrapped JSON. Each was corrected before the focused suite reached 9/9; the
CLI plus existing CLI regression suite reached 11/11.

The rejection remediation added RED/GREEN coverage for the logical store
catalog, locked quarantine validation and rollback, hostile backup manifests,
failed final publication, coordinator DuckDB snapshots, manifest identity
substitution, alias/mode-aware fail-closed architecture scanning, catalog lock
mapping, central secret-safe Agent storage payloads, every CLI outcome, and
destructive migration approval. The final focused persistence/Agent/CLI run was
151 passed.

## Scope and limitations

Backup v1 is a verified local copy, not off-device disaster recovery. It excludes
cache, temp, locks, and prior backups. Restore remains a staged operator action.
No SQLite, control service, cloud, or distributed migration was added. Explicit
quarantine does not replace the specialized Tushare ledger recovery workflow.

## Final verification

Fresh rejection-remediation `make check` completed successfully: Ruff passed,
mypy reported no issues across 201 source files, and all 627 tests passed with
76 existing dependency/deprecation warnings. `git diff --check` was clean.

## Commits

- `748419f feat(storage): add local operations and persistence enforcement`
- `ce372d1 feat(cli): expose storage operations commands`
- `aef4db6 fix(storage): enforce backup writer barrier`
- `11530eb fix(storage): drive operations from logical store catalog`
- `1926966 fix(storage): make quarantine validation locked and failure-safe`
- `ffea285 fix(storage): harden persistence architecture scanner`
- `23ebf9a fix(agent): surface secret-safe storage health payloads`
- `531c519 fix(storage): publish strictly verified backup snapshots`
- `f6955f1 fix(storage): map lock diagnostics to catalog resources`
- `36f2cd2 fix(storage): bind artifact manifests to identity filenames`
- `7c98e4f fix(cli): cover storage command outcomes consistently`
- `f98f180 test(storage): enforce destructive migration approval`
- `9948ce4 fix(agent): type storage health payload boundary`
