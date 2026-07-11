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
