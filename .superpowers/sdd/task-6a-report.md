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
- Generic `AtomicFileStore.append_jsonl` retains its compact-byte contract;
  `AuditJsonlStore` explicitly selects legacy spaced JSON formatting required by
  existing Agent audit consumers without changing locking or write count.
- Rotation size check, monotonic numbered-generation rename, and boundary append
  share one resource lock. Retention defaults to unlimited, and readers/verifier
  traverse every generation in order before the active file.
- Flush is intrinsic to the descriptor write and fsync is configurable through
  `Settings.audit_fsync`; rotation size is `Settings.audit_rotation_bytes`.
- Verification reports an incomplete final line as a truncated tail and reports
  malformed complete lines as line-numbered mid-file corruption.
- Agent audit recursively scrubs exact credential keys and known credential
  patterns while preserving benign telemetry such as `token_count`,
  `token_budget`, and ordinary text containing the word “token”.
  The normalized key classifier covers Settings and provider variants including
  `qmt_gateway_hmac_secret`, `hmac_secret`, `api_secret`, and `access_key`, plus
  hyphenated/camel-case suffix variants and credential assignment strings.
  Keys are canonicalized to lowercase alphanumeric names so acronym forms such
  as `providerAPISecret` and `qmtGatewayHMACSecret` cannot bypass suffix checks;
  assignment identifiers use the same classifier while benign assignments and
  token telemetry remain visible.
  Both `=` and `:` assignments are parsed, including JSON-like quoted keys and
  values; only the credential value is replaced so surrounding audit text is
  preserved.
- Added `ContentAddressedCache`: canonical JSON SHA-256 key, schema/TTL envelope,
  validated atomic writes, hit/miss/expiry/corruption/write-failure metrics, and
  structured warnings. Corruption, invalidation, warning-sink, and write faults
  never block research.
- Runtime creates one cache and injects it through `AgentToolDependencies` and
  context-scoped factor/strategy tool execution. Module-level and CWD cache roots
  were removed.
- Cache reads, writes, and conditional invalidation share the canonical resource
  lock. Factor freshness rejection uses the cache API; optional cache absence
  degrades to an ordinary miss and never changes tool availability.

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
- Review RED: six forced rotations retained only the last two events; a
  multiprocess rotation test retained 2/80. Optional cache absence returned
  `NOT_IMPLEMENTED`; factor freshness bypassed cache metrics/API; and a writer
  completed while a corrupt reader was paused before invalidation.
- GREEN: focused audit/cache/adapter/tool suites: 67 passed.

Covered multi-process audit append, repeated and multiprocess rotation, half-tail,
mid-file corruption, fsync policy, exact secret scrub, backward read, CWD
independence, cache TTL/hit/miss, corrupt/freshness invalidation, unlink and write
fault preservation, optional-cache behavior, and controlled concurrent replacement.

## Verification

```text
uv run pytest tests/unit/persistence/test_audit_jsonl.py \
  tests/unit/persistence/test_content_cache.py \
  tests/unit/agent/test_factor_cache.py \
  tests/unit/agent/test_tool_registry.py \
  tests/unit/agent/test_factor_workflow.py \
  tests/unit/agent/test_strategy_workflow.py -q
67 passed

uv run ruff check src tests/unit/persistence/test_audit_jsonl.py \
  tests/unit/persistence/test_content_cache.py tests/unit/agent/test_factor_cache.py \
  tests/unit/agent/test_factor_workflow.py
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
- Rotation generations are unlimited by default because audit JSONL is source of
  truth; no implicit retention path silently deletes prior events.

## Commit

`fix(persistence): harden audit and disposable caches`

Review correction: `fix(persistence): preserve audit generations and cache semantics`

Final scrub correction: `fix(audit): cover normalized credential key variants`

Acronym scrub correction: `fix(audit): share canonical credential classification`

Assignment scrub correction: `fix(audit): redact colon and quoted assignments`

JSONL formatting correction: `fix(persistence): preserve generic JSONL encoding`
