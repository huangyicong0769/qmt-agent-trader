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

## Scope and limitations

Backup v1 is a verified local copy, not off-device disaster recovery. It excludes
cache, temp, locks, and prior backups. Restore remains a staged operator action.
No SQLite, control service, cloud, or distributed migration was added. Explicit
quarantine does not replace the specialized Tushare ledger recovery workflow.

## Commits

- `748419f feat(storage): add local operations and persistence enforcement`
- `ce372d1 feat(cli): expose storage operations commands`
- `aef4db6 fix(storage): enforce backup writer barrier`
