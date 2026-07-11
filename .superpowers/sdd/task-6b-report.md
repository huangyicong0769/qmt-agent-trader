# Task 6B Report — Immutable Governed Artifacts

## Scope

Implemented only the Phase 5 order-plan, approval, report, and generated
factor/strategy artifact slice. Audit/cache work and Phase 6 were not changed.

## Inventory and migration

- Added a shared `ArtifactStore` over the approved `AtomicFileStore` and
  `LockManager`. It accepts only contained relative paths, publishes exact bytes
  create-only, and writes a deterministic manifest keyed by artifact id.
- Manifest fields include schema version, artifact id/type, created timestamp,
  producer, SHA-256, byte length, relative path, and related run, strategy, and
  factor ids.
- Order plans are immutable by `order_plan_id`; verified reads reject tampering.
  Risk-check and paper-acceptance state is appended to a separate JSONL event
  stream. CLI execution reads the original plan plus its prior event history.
- Strategy approval YAML remains human-readable. `{strategy_id, version}` is
  create-only, and storage preserves caller-supplied `approved_by` and
  `approved_at` exactly. Trusted workflow and LLM permission gates are unchanged.
- Research JSON, backtest JSON, strategy-backtest JSON, and generated Markdown
  reports now use immutable run/report identities and manifests. Existing
  `path`/`report_path` keys remain; optional `manifest_path`/`storage_status` are
  additive where exposed.
- `CodeSandbox` still performs containment and static scanning before storage.
  Factor and strategy implementation/test/spec writers now add explicit run and
  factor/strategy provenance and cannot silently replace the same candidate.

No database migration was added. Files and manifests remain the source of truth;
a database may index them later but cannot replace them.

## Failure and recovery behavior

- Same artifact id or physical target is a structured conflict, including a
  manifest-only residual identity.
- Faults before content publication leave no official artifact. A simulated
  fault after content publication but before manifest publication rolls the
  content back and never reports success.
- Read verification reports missing content and hash mismatch. Repository
  diagnostics additionally report orphan content and invalid manifests; a real
  process crash that defeats rollback is therefore explicit rather than hidden.
- Concurrent create has exactly one winner under the shared resource lock.

## TDD RED / GREEN evidence

- RED: artifact tests failed collection because `persistence.artifacts` did not
  exist. GREEN: 10 shared-store tests pass.
- RED: domain tests failed because order-plan event APIs, immutable approval
  manifests, and sandbox manifests did not exist. GREEN: order-plan, approval,
  and sandbox group passes (23 tests in the focused run; 38 with report services).
- RED: research/backtest receipts had no manifests or structured storage status.
  GREEN: JSON report group passes (5 tests).
- RED: Markdown and strategy-backtest report assertions found no manifest.
  GREEN: focused report integration passes (2 tests).
- Review RED: a shared report directory contained multiple valid manifests, so a
  count-based assertion was invalid. It now matches the report by relative path
  and run id; the focused test passes.

## Verification

```text
Artifact/order/approval/report/sandbox focused: 38 passed
Factor authoring split: 3 + 3 + 1 + 2 = 9 passed
Factor workflow split: 3 + 3 + 4 + 5 + 4 = 19 passed
Strategy workflow split: 6 + 3 + 4 = 13 passed
Meta/contracts/permission/tool registry: 49 passed
Backtest factor/snapshot/rolling: 11 passed, one test-assertion RED corrected;
focused correction: 1 passed
uv run ruff check src ...: All checks passed
uv run mypy src: Success, 198 source files
git diff --check: clean
```

The split runs avoid a long combined pytest invocation that stayed alive after
partial dot output; stale processes were terminated before serial verification.
Only existing `streamable_http_client` deprecation warnings were emitted.

## Compatibility and governance

- Existing domain paths and tool result keys remain usable; generated Markdown
  now adds a unique report-id directory rather than overwriting an experiment
  filename.
- Generated candidates remain research-only/review-required. Approval-required
  tools remain unavailable to autonomous LLM calls, and live order submission
  remains forbidden to the LLM.
- No audit/cache behavior, factor math, strategy math, UI, cloud backup, or Phase
  6 CLI was introduced.

## Commits

- `8f099ec feat(persistence): govern immutable trading artifacts`
- Documentation/report commit: recorded separately after this report update.

## Review remediation

The rejected review was addressed in correction commit
`e2e4d6c fix(persistence): preserve and verify governed artifacts`:

- Existing order-plan JSON and approval YAML are validated and adopted under the
  artifact lock without rewriting bytes, paths, ids, or human approval fields.
  Adoption is idempotent; malformed legacy governance files raise structured
  storage validation errors and receive no manifest.
- Manifest reads bind the requested artifact id to the manifest payload and bind
  the expected repository/file path to `relative_path`, preventing manifest
  substitution.
- Research/backtest compare and research-report search consumers adopt valid
  legacy reports and verify governed hashes. Tampered governed reports are
  rejected or represented as blocked evidence with an explicit verification
  reason.
- Generated factor, strategy, and meta-tool code now includes logical version
  plus `ToolContext.run_id` in artifact identity and physical path. Two runs of
  the same logical candidate are both retained and discoverable.
- Runtime and registry composition explicitly derive the generated root from the
  injected Settings/PersistencePaths project root; injected executions no longer
  fall back to global settings.
- Approval CLI reruns recover after a simulated failure between artifact publish
  and registry attachment. The verified original approval is reused, then attach
  and trusted status update complete; signer and timestamp are unchanged.
- Order-plan events now carry `schema_version` and `event_id`; append and reads
  share the same resource lock, malformed complete records fail closed, and only
  an incomplete final tail is ignored.

Review correction evidence:

```text
Legacy/binding/event/approval slice: 22 passed, then corrupt-file additions: 12 passed
Core/domain/CLI/runtime/sandbox: 58 passed
Run-version/custom-root focused: 4 passed
Governed report compare/adoption: 2 passed
Version/search/contracts focused: 8 passed
Affected strategy candidate workflows: 5 passed
Meta/report/contracts: 10 passed
Ruff: All checks passed
mypy: Success, 198 source files
git diff --check: clean
```

## Second rereview remediation

Correction commit `8453114 fix(persistence): make artifact adoption race safe`
closes the remaining rereview findings:

- `ArtifactStore.adopt` now rereads, optionally matches expected bytes, validates,
  hashes, and manifests the exact same byte string while holding the artifact-root
  lock. It returns those adopted bytes; order, approval, research, backtest, and
  research-search callers parse only that locked result. A stale expectation is a
  structured retryable conflict rather than a manifest for changed content.
- Approval adoption validates that parsed `strategy_id` and `strategy_version`
  reconstruct the exact expected filename before any manifest is published.
- Generated meta tests use `importlib` with the supplied `code_path`, so tests
  execute the exact run-version implementation instead of an obsolete package
  path.
- `OrderPlanEvent.schema_version` is `Literal[1]`; unknown versions fail model
  validation.
- Factor, strategy, and meta run directories use one shared slug plus stable hash
  encoder. Distinct ids such as `a/b` and `a?b` cannot collapse to one path.

Second rereview evidence:

```text
Five targeted REDs initially failed; all five pass after correction.
Artifact/order/approval/research/backtest/meta/search: 42 passed
Ruff: All checks passed
mypy: Success, 198 source files
git diff --check: clean
```
