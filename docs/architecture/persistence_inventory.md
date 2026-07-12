# Persistence boundary inventory

- Status: Phase 0 architecture baseline
- Scope: production code on `codex/persistence-storage-refactor-v2` as inspected on 2026-07-10
- Decision records: [control store](adr-control-store.md), [artifact store](adr-artifact-store.md)

## Purpose and classification

This inventory records every production persistence boundary found by searching for direct file writes, append mode, Parquet writes, DuckDB connections, and DuckDB DDL/DML. It is a migration map, not a claim that the current boundaries are safe.

Source owners in the tables are exact project-relative paths beneath `src/qmt_agent_trader/`; physical paths are templates relative to `Settings.project_root` unless explicitly marked CWD-relative.

- **Analytical Data Store (ADS):** tabular market/research data, derived tables, and the DuckDB analytical catalog.
- **Control Plane Store (CPS):** mutable application state that coordinates work or records current status.
- **Artifact Store (AS):** human-reviewable or audit evidence, including immutable plans/reports and generated code.

Unless a row says otherwise, there is no backup, restore, quarantine, manifest, hash, schema migration, or revision history. `Settings.project_root` plus `data_dir`/`log_dir` injects the principal data and log roots, but several module-level relative paths still depend on the process working directory.

## Analytical data and catalog

| Domain | Owner and exact write boundary | Source of truth and physical path | Format; mutability; access pattern | Current coordination, metadata, and recovery | Known failure mode and later migration target |
| --- | --- | --- | --- | --- | --- |
| Raw provider datasets | `data/providers/tushare/fetcher.py:TushareFetcher.execute_plan` -> `DataLake.write_incremental_dataset`; `cli/main.py:data_migrate_new_layout` -> `DataLake.write_incremental_parquet`, then conditional legacy `Path.unlink` | Provider response merged into `{project_root}/{data_dir}/lake/raw/{raw_dataset_name}.parquet`; registry names normally map `dataset_id` to `tushare/<api_name>`. The CLI migration merges stable legacy files and matching `{old_name}_{YYYYMMDD}_{YYYYMMDD}.parquet` batches into `raw/tushare/*.parquet`, then deletes each source unless `--keep-legacy`. | Parquet; mutable keyed snapshot; read-merge-deduplicate-sort-replace. CLI cleanup is destructive after target publication. | Target writes use per-dataset `FileLock(<parquet>.lock)` plus validated temp Parquet, file `fsync`, `os.replace`, and directory `fsync`; provider keys come from endpoint registry; fetch checksum is stored separately. Legacy source reads/unlinks have no source lock, manifest, or cross-file transaction. | Concurrent target writers are serialized only per target Parquet path. A crash during the CLI unlink loop can leave only a subset of legacy sources, and a concurrent legacy-source writer can race between read and unlink; neither target publication nor multi-source cleanup is transactional with catalog/fetch state. Remains ADS behind a data-store interface. |
| Silver tables | `data/table_builder.py:DataTableBuilder.build` | Raw lake inputs are derivation source; `data/lake/silver/{security_master,trade_calendar,daily_market,index_daily,financial_reports_wide,financial_current_wide,corporate_actions,macro_series}.parquet` | Parquet; mutable keyed materializations; rebuild/merge/read | Same per-file lock and atomic writer; `updated_at` column; key policy in `_silver_keys`; no build version or lineage manifest | A Parquet update can commit while view registration fails, and changed transform logic has no explicit schema/build version. Remains ADS. |
| Gold factor outputs | `factors/service.py:compute_factor_to_lake` -> `DataLake.write_parquet` | Saved factor implementation plus lake inputs; `data/lake/gold/factor_{name}_{YYYYMMDD}.parquet` | Parquet; date-named derived output, replaceable; full-file write/read | Atomic validated replacement but **no dataset lock** in `write_parquet`; name/date is the only version signal; no backup | Same-name concurrent computation can last-writer-win; replacement can diverge from catalog registration. Remains ADS; immutable-by-generation naming should be preferred later. |
| Analytical DuckDB catalog | `data/storage.py:DataLake.connect`, `register_parquet`, `register_dataset_id`; invoked after raw/silver/gold writes | `data/qmt_agent_trader.duckdb`; Parquet is data truth, DuckDB views are the query catalog | DuckDB views created/replaced with `CREATE OR REPLACE VIEW ... read_parquet(...)`; mutable catalog | View writes use the shared `DatabaseCoordinator` database lock and explicit transaction; view name validation remains; schema migrations are recorded separately | File-relative Parquet registrations and a failed post-write registration can still leave catalog/data disagreement. Remains ADS behind the global write coordinator. |
| Generic fetch state/current rows | `data/storage.py:DataLake.ensure_fetch_tables`, `record_fetch_metadata`, `record_fetch_result` | `data/qmt_agent_trader.duckdb`, tables `data_fetch_state_v2`, `data_fetch_events_v2`, legacy `data_fetch_state`, `data_fetch_events` | DuckDB; current-state tables have declared primary keys and use upsert; event tables are append-only | Startup migrations create/upgrade schemas; old no-PK state tables are deterministically deduplicated and rebuilt; each current-state upsert and event insert shares one coordinator transaction | State/event failure now rolls back atomically and process writers serialize. The legacy pair remains only as a compatibility contract pending removal; see the deprecation note below. |
| Tushare usage/fetch ledger | `data/providers/tushare/quota.py:TushareUsageLedger.append`, `record_history_reset`; reads in usage/cache methods | `data/qmt_agent_trader.duckdb`, tables `tushare_usage_events_v1`, `tushare_usage_state_v1`, `tushare_usage_migrations_v1` | DuckDB; usage events create-only by `request_id`; state mutable; migration audit append/update | All writes use the shared `DatabaseCoordinator`; startup migrations own DDL, and normal reads perform readiness assertions only; structured corruption error still points to repair | CPS semantics and table names remain unchanged. Legacy Parquet import/archive/quarantine is scheduled by explicit initialization rather than a quota read. |
| Legacy Tushare ledger migration/audit | `data/providers/tushare/ledger_migration.py:migrate_legacy_usage_ledger`, `_finalize_migration`, `_finalize_archived_pending_migrations` | Legacy `data/lake/metadata/tushare_usage_ledger.parquet`; archive `.../metadata/archive/tushare_usage_ledger.migrated.<UTC>.parquet`; migration table in DuckDB | Parquet input becomes immutable archive; DuckDB migration row records pending/final state | `migrate_legacy_usage_ledger` holds the Tushare mutation lock across legacy read/validation, transactional import and pending marker, `os.replace` archive, and its call to `_finalize_migration`; the helper opens a connection but does not reacquire the already-held lock. Startup `_finalize_archived_pending_migrations` acquires the lock itself; request-id conflict is idempotent. | Crash after DB commit is recoverable through pending state; filesystem archive has no directory `fsync`, and archive replacement plus final status update are not one filesystem/database transaction despite sharing the outer lock. Preserve this completed recovery protocol and later use the global coordinator. |
| Corrupt legacy Tushare quarantine | `data/providers/tushare/ledger_migration.py:repair_tushare_usage_ledger`, `_quarantine_corrupt_legacy`, `_atomic_write_json` | Corrupt legacy Parquet moved to `.../metadata/corrupt/tushare_usage_ledger.<UTC>.parquet`; JSON sidecar adjacent; history-reset row in DuckDB | Immutable quarantined bytes plus JSON evidence | File lock; SHA-256 and size; temp JSON `fsync` + `os.replace`; DuckDB reset marker; structured error/warning | Reset marker can commit before sidecar/move fails; sidecar replacement lacks directory `fsync`; no external backup. Keep quarantine semantics, add manifest/store API later. |

### Analytical invariants to preserve

The refactor must preserve the validated-temp/atomic-replace Parquet path in `data/atomic_io.py:atomic_write_parquet`, including row-group validation and directory `fsync`. It must also preserve all three Tushare table names, legacy import/archive/resume behavior, quarantine evidence, structured `TushareUsageLedgerCorruptError`, request-id idempotency, redaction, and quota-history warnings.

### Legacy fetch metadata deprecation

As of the Phase 2 implementation on 2026-07-11, repository production code has no callers of `record_fetch_result`, `fetch_state`, or `fetch_events`; only compatibility tests exercise `data_fetch_state` and `data_fetch_events`. The tables and methods remain supported so existing local databases and external callers are not broken. New provider code must use the v2 metadata contract. Removal requires a later explicit migration after downstream caller inventory and a release deprecation window; it is not part of this refactor.

## Control plane state

| Domain | Owner and exact write boundary | Source of truth and physical path | Format; mutability; access pattern | Current coordination, metadata, and recovery | Known failure mode and later migration target |
| --- | --- | --- | --- | --- | --- |
| Factor registry | `factors/registry.py:FactorRegistry._persist_file_registry` | `data/factors/registry.json` when production tools pass `lake.root.parent / "factors"` | JSON object `{version: 1, factors: [...]}`; mutable whole-file registry | Direct `Path.write_text`; no lock/atomic swap/backup; explicit version `1` | Torn/truncated write; concurrent save/update loses entries; reader may see invalid JSON. CPS repository with revision/CAS and atomic locked writes. |
| Strategy registry | `strategy/registry.py:StrategyRegistry._persist_file_registry` | `data/strategies/registry.json` | JSON object `{version: 1, strategies: [...]}`; mutable whole-file registry | Direct write; no lock/atomic swap/backup; explicit version `1`; approval transition rules live above storage | Same corruption/lost-update risk; registry and attached report/code/approval references are not transactional. CPS repository, preserving approval state machine. |
| Universe registry | `universe/registry.py:UniverseRegistry.save`; root from `for_lake` or `registry_root_from_payload` | Normally `data/universes/registry/{sanitized_universe_id}.json`; injectable payload may select any `Path` | One JSON document per universe; mutable overwrite; model carries research/approval flags but no storage envelope version | Direct write; no lock/atomic swap/backup; only `/` is sanitized in id; caller-controlled root | Concurrent overwrite and partial JSON; unsafe/ambiguous root injection and insufficient filename normalization. CPS repository with validated injected root and revisions. |
| Experiment state | `agent/experiment_store.py:ExperimentStore._write`; created by runtime/CLI/routes | `data/experiments/{experiment_id}.json` | JSON per experiment; mutable read-modify-rewrite; `updated_at`, model schema but no storage version | Direct write, no lock/atomic swap/backup | Workflow/tool updates can overwrite each other; interrupted write loses the only record. CPS repository with revision/CAS. (The module docstring's “JSONL-backed” claim is stale; implementation is JSON.) |
| Session todo state | `agent/todos.py:TodoListStore._write` | `data/todos/{sha256(session_id)[:16]}.json` | JSON per session; mutable replacement/deletion of completed items; model timestamps | Direct write, no lock/atomic swap/backup; hashed filename avoids raw id in path but is not a revision | Concurrent tool calls lose list edits; truncation loses progress. CPS repository with session key, revision, and locked atomic update. |
| Chat sessions | `web/ui/pages/chat.py:_ChatSession.save`, `close_session`; module constant `SESSIONS_DIR` | **CWD-relative** `sessions/{sid}.json`; deletion with `unlink` | JSON conversation snapshot; mutable whole-file rewrite and destructive delete | Directory created at import; no lock/atomic swap/backup/version; load silently skips malformed JSON | Two web processes or overlapping saves lose messages; partial file silently disappears from UI; CWD changes split state. CPS session repository with injected project-root path and revisioned writes. |
| Validation/backtest cache | Factor and strategy tool cache helpers; runtime injects `ContentAddressedCache` through `AgentToolDependencies` | Canonical `PersistencePaths.cache_root/{factor-validation,backtest}/{sha256}.json` | Disposable JSON envelope; deterministic content key, schema version and configurable TTL | Validated atomic replace; corrupt/expired entries auto-invalidate with structured warning and metrics; failures do not block research | Existing tool hit/miss behavior is preserved; cache remains excluded from durable backup. |

## Reviewable and audit artifacts

| Domain | Owner and exact write boundary | Source of truth and physical path | Format; mutability; access pattern | Current coordination, metadata, and recovery | Known failure mode and later migration target |
| --- | --- | --- | --- | --- | --- |
| Agent tool audit | `agent/audit.py:AuditLogger`; runtime injects canonical path and shared storage dependencies | `PersistencePaths.audit_root/agent_tool_calls.jsonl` | Schema-versioned JSONL source of truth; legacy rows readable; recursive exact-pattern secret scrub | Single locked `os.write`, configurable fsync, unlimited numbered rotation generations; verifier covers every generation plus active file | Multi-process records cannot interleave and repeated rotation preserves every event. |
| Core/CLI trade audit stream | `core/audit.py:AuditLogger`; `cli/main.py:_audit_logger` | `PersistencePaths.audit_root/trade.jsonl` by default | Schema-versioned JSONL source of truth with backward-compatible reader | Shared `AuditJsonlStore`, canonical `LockManager`, configurable fsync/rotation and read-only verifier | Public append/read contracts remain unchanged. |
| Strategy approvals | `strategy/approval.py:write_approval_file`; CLI injects `PersistencePaths.approvals_root` | `{approvals_root}/{strategy_id}_{strategy_version}.approval.yaml` plus `.manifests/{sha256(artifact_id)}.json` | Human-readable YAML governance artifact; immutable decision per strategy version; exact signer and timestamp bytes retained | Shared `ArtifactStore`; legacy YAML is validated then adopted without rewriting; governed reads verify hash/id/path. Same-content CLI retries resume registry attachment/status after partial failure. | A different decision for the same strategy/version conflicts. Storage does not create or alter `approved_by`, `approved_at`, or trading permissions. |
| Order plans and governance events | `services/order_plan_service.py:save_order_plan`, `append_order_plan_event`; CLI injects `PersistencePaths.order_plans_root` | `{order_plans_root}/{order_plan_id}.json`, manifest, and separately cataloged `order_plan_events` at `.events/{sha256(order_plan_id)}.jsonl` | Immutable JSON plan plus schema-versioned append-only lifecycle events | All governed APIs require the canonical artifact store. Event append/read uses canonical locking; runtime and `storage verify` share validation for complete tails, schema, unique event ids, one order-plan id, and filename/id binding. | Any truncated tail, malformed record, mixed identity, duplicate event id, or tampered plan fails closed. Main-plan quarantine moves content, manifest, and its event stream as one governance unit. |
| Research JSON reports | `services/research_report_service.py:save_research_report` | Injected reports root `{research_run_id}.json` plus deterministic manifest | Immutable JSON carrying research-only, approval and decision-boundary metadata | Shared path-safe create-only `ArtifactStore`; exact-byte SHA-256 and optional structured `storage_status` receipt | Duplicate run identity conflicts; partial/faulted publication is not returned as saved. |
| Backtest JSON reports | `backtest/service.py:run_backtest_report`; `strategy/execution_adapter.py:run_strategy_backtest` | Injected report root `{run_id}.json` plus deterministic manifest | Immutable run artifact with run/strategy/factor provenance | Shared create-only `ArtifactStore`; exact-byte hash, verified manifest, shared data-lake lock where available | Duplicate run identity cannot replace reviewed evidence; manifest diagnostics expose missing or hash-mismatched files. |
| Research Markdown reports | `agent/tools/strategy_tools.py:_generate_research_report` | Sandbox `generated/reports/{experiment_id}/{report_id}.md`, or injected research root with the same run identity | Human-reviewable immutable report generation; each invocation receives a distinct report id | Shared `ArtifactStore`; create-only exact bytes, manifest, returned `report_path`, `manifest_path`, and optional `storage_status` | Re-running creates a new report artifact rather than rewriting prior narrative. |
| Generated factor/strategy/tool code | `agent/sandbox.py:CodeSandbox.write_candidate_file`, called by factor/strategy/meta tools | Settings-injected generated root; candidate/version/run path plus `.manifests/{sha256(artifact_id)}.json` | Review-required immutable candidate versions; repeated logical factor/strategy/tool generation creates a new run-specific path | Existing containment/static scan precedes shared create-only `ArtifactStore`; manifest identity includes run and logical factor/strategy/tool version; discovery lists retained runs | A single run cannot overwrite itself, while legitimate later runs remain side by side. Live-trading and approval restrictions are unchanged. |
| Tool-registration proposals | `agent/tools/meta_tools.py:_propose_tool_registration` | Normally sandbox `.../generated/tools/tool_proposal_{candidate_id}.json`; fallback **CWD-relative** `proposals/` | JSON review artifact; intended immutable per candidate | Direct overwrite; no lock/create-only/hash/manifest | Proposal evidence can be replaced after scoring. AS create-only artifact tied to generated-code manifest. |
| Oversized subprocess payload spill | `agent/tool_registry.py:_prepare_process_payload`, `_process_payload_path` | **CWD-relative** `reports/tool_payloads/{sanitized_run_id}_{sanitized_tool}.json` | JSON transient transport artifact; mutable/reused by run/tool key | Direct write; no lock/atomic swap/expiry/backup; filename sanitization | Parallel same tool/run calls overwrite payloads and can resolve the wrong result; partial payload fails process response. CPS transient blob/cache API with unique call id and atomic create. |

## Module-relative paths requiring explicit migration

These paths bypass `Settings.resolved_*` and must be injected through the future store interfaces instead of being resolved from `cwd` or accepted as unrestricted payload strings:

- `web/ui/pages/chat.py:SESSIONS_DIR` -> `sessions/`.
- Validation/backtest caches now use injected `PersistencePaths.cache_root`; no production cache root is derived from CWD.
- `agent/tool_registry.py:_process_payload_path` -> `reports/tool_payloads/`.
- `agent/tools/strategy_tools.py` fallback reports -> `reports/research/`.
- `agent/tools/meta_tools.py` fallback proposals -> `proposals/`.
- `cli/main.py` approval and order-plan commands -> `approvals/` and `order_plans/`.
- `services/order_plan_service.py:load_order_plan` default -> `order_plans/`.
- `universe/registry.py:registry_root_from_payload` fallback -> `data/universes/registry/`, and its caller-supplied `registry_root` requires root-policy validation.
- `agent/sandbox.py:CodeSandbox` default is derived from the source tree rather than `project_root`; preserve path containment but inject the artifact root.

## Direct-write migration checklist

The following direct production boundaries must later call shared storage APIs: every `Path.write_text` listed above; both JSONL append implementations; generated-code `write_text`; cache deletion/read-repair; and the migration sidecar writer (which is already atomic but should share the manifest/quarantine API). No production `Path.write_bytes` boundary exists in the inspected tree. `DataFrame.to_parquet` occurs only inside `data/atomic_io.py:atomic_write_parquet`; that implementation is preserved and wrapped, not replaced with a weaker writer. `duckdb.connect` occurs in `DataLake.connect` and `TushareUsageLedger.connect`; both must be delegated to the single coordinator without changing existing tables or semantics.

## Out-of-scope executable scripts

`scripts/replay_session9_smart_beta_real_agent.py` and `scripts/profile_research_tools.py` also use direct `write_text` for replay/profiling outputs. They are operator/development scripts rather than imported production boundaries, so they are not store sources of truth. When those scripts are retained, they should consume the same artifact API or be explicitly marked disposable; this Phase 0 task does not modify them.

## Evidence method

The baseline was produced from current code using searches for `write_text`, `write_bytes`, append-mode `open`, `to_parquet`, `duckdb.connect`, DuckDB DDL/DML, `os.replace`, and all callers of the registry/store constructors and lake writers. No persistence behavior in this document is inferred from the refactor proposal alone.

## Phase 6 operations reconciliation

`StoreCatalog.canonical(PersistencePaths)` is now the single executable inventory
boundary. `qmt-agent storage inventory` emits one structured entry per real
logical store: DuckDB, raw/silver/gold/metadata lake layers, factor/strategy
registries, todos, experiments, sessions, canonical and legacy universes,
approvals, order plans/events, separate governed `reports/backtests` and
`reports/research`, audit streams, and governed generated code at the production
composition root `src/qmt_agent_trader/agent/generated`. Configured
`reports/cache`, `reports/tool_payloads`, temp, locks, backups, and quarantine are
excluded by construction.
Each entry includes type, exact path, owner, truth role, schema version,
mutability, lock resource, backup policy, and current presence.
`StorageOperations` owns verify, migration delegation, local backup verification,
lock diagnostics, and explicit quarantine; existing `data validate` and Tushare
repair commands remain intact.

Every production mutation owner uses the canonical `PersistencePaths.locks_root`
gate. Default ArtifactStore construction, DataLake Parquet publication, audit
appenders, generated-code/proposal writes, report writers, and DuckDB writes all
cooperate with the exclusive backup barrier. Architecture enforcement rejects
new CWD persistence roots and private `.locks`/`.artifact-locks` managers outside
the narrow configuration-owner allowlist.

An AST architecture test rejects new `Path.write_text`, `DataFrame.to_parquet`,
direct or aliased `duckdb.connect`, and positional/keyword append-mode built-in
or `Path.open` calls. Invalid or unreadable production Python fails closed. The narrow
allowlist covers only `AtomicFileStore.write_parquet` and
`DatabaseCoordinator._open_connection`, the infrastructure owners documented
above. Core tool-proposal and process-payload writes were migrated to
`AtomicFileStore` rather than exempted.
