# Task 6A Report — Audit and Cache Hardening

## Scope

Implemented only the Phase 5 audit JSONL and disposable factor-validation/backtest
cache portions. Order plans, approvals, reports, generated code, artifacts, and
Phase 6 CLI were not changed.

## Inventory and ownership

- Agent tool audit: `agent.audit.AuditLogger`, canonical
  `PersistencePaths.audit_root/agent_tool_calls.jsonl`.
- Core trade audit: `core.audit.AuditLogger`, canonical
  `PersistencePaths.audit_root/trade.jsonl`.
- Factor validation cache: `agent.tools.cache`, injected
  `PersistencePaths.cache_root/factor-validation`.
- Strategy backtest cache: `agent.tools.strategy_tools`, injected
  `PersistencePaths.cache_root/backtest`.
- Tushare equivalence/request cache remains control-plane ledger state and was
  intentionally not treated as a disposable file cache.

The architecture inventory was updated with the implemented owners, paths,
coordination, metadata, and recovery policy.

## Implementation

- Added `AuditJsonlStore` and reusable read-only `verify_audit_jsonl`.
- Audit rows carry `schema_version=2`; legacy unversioned rows remain readable.
- Shared canonical `LockManager` plus `AtomicFileStore` serialize multi-process
  append and rotation. Each row is encoded once and emitted by one `os.write`.
- Rotation size check, rename, and boundary append share one resource lock.
- Flush is intrinsic to the descriptor write and fsync is configurable through
  `Settings.audit_fsync`; rotation size is `Settings.audit_rotation_bytes`.
- Verification reports an incomplete final line as a truncated tail and reports
  malformed complete lines as line-numbered mid-file corruption.
- Agent audit retains existing error-message scrub and recursively scrubs secret
  keys/suspicious secret-bearing strings across the persisted record.
- Added `ContentAddressedCache`: canonical JSON SHA-256 key, schema/TTL envelope,
  validated atomic writes, hit/miss/expiry/corruption/write-failure metrics, and
  structured warnings. Corruption, invalidation, warning-sink, and write faults
  never block research.
- Runtime creates one cache and injects it through `AgentToolDependencies` and
  context-scoped factor/strategy tool execution. Module-level and CWD cache roots
  were removed.

## Compatibility and migrations

- Audit public append/read methods, tool audit fields, status values, hashes, and
  JSON spacing relied upon by callers remain compatible.
- Existing unversioned audit rows are read without migration.
- Existing cache files use the former filenames/envelope and naturally miss;
  caches are disposable, so no data migration is required.
- Factor/backtest `cache_hit` behavior remains unchanged.

## RED / GREEN evidence

- RED: new focused tests initially failed collection because
  `persistence.audit` and `persistence.cache` did not exist.
- RED: adapter test then failed because public `AuditLogger` did not accept the
  injected atomic store.
- GREEN: focused audit/cache/adapter/tool suites: 40 passed.

Covered multi-process audit append, half-tail, mid-file corruption, rotation
boundary, fsync policy, secret scrub, backward read, CWD independence, cache
TTL/hit/miss, corrupt invalidation, fault preservation, and concurrent writes.

## Verification

```text
uv run pytest tests/unit/persistence/test_audit_jsonl.py \
  tests/unit/persistence/test_content_cache.py \
  tests/unit/agent/test_factor_cache.py \
  tests/unit/agent/test_tool_registry.py \
  tests/unit/agent/test_factor_workflow.py \
  tests/unit/agent/test_strategy_workflow.py -q
40 passed

uv run ruff check src tests/unit/persistence/test_audit_jsonl.py \
  tests/unit/persistence/test_content_cache.py tests/unit/agent/test_factor_cache.py
All checks passed

uv run mypy src
Success: no issues found in 197 source files

git diff --check
clean
```

## Failure recovery and concerns

- Audit append failure truncates back to the locked original length; rollback
  failure remains a structured `StorageAppendRollbackError`.
- Cache corruption is deleted with warning/metric; failed replacement preserves
  the previous valid entry and degrades to a miss.
- Rotation currently retains one `.1` generation, matching the bounded safe
  rotation implemented here. Broader retention/export policy belongs to later
  artifact/operations work.

## Commit

`fix(persistence): harden audit and disposable caches`
