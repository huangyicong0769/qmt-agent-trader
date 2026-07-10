# Task 5 report — Versioned mutable-state repositories

## Result

Implemented a shared one-record-per-file repository boundary and migrated Todo,
Experiment, Chat Session, and Universe state in the required order. Records use
canonical resolved paths, per-resource inter-process locks, atomic JSON replace,
schema version 2, monotonic revision, updated timestamp, deterministic SHA-256
content hash, reread verification, structured errors, diagnostic listing, and
explicit quarantine.

## RED / GREEN evidence

- Chat RED: removing the module-global `_sessions` source produced seven focused
  API failures. Tests moved to repository injection, not a production test seam.
- Chat RED: an intermediate cached repository caused list contamination
  (`expected 1 session, found 2`). Repository injection removed it.
- GREEN: focused Todo, Experiment, Chat, Universe and seam suites: `39 passed`.
- GREEN: focused Ruff checks pass.
- Spawn-process tests cover Todo mutations and Experiment artifact appends;
  additional tests cover corrupt diagnostics, quarantine, and unsafe Universe IDs.

The new concurrency/corruption tests passed on their first run after the behavior
was implemented; they are regression evidence, not claimed as RED evidence.

## Migration and compatibility

- Valid legacy unversioned records migrate lazily and idempotently under lock,
  retaining IDs and content.
- Todo retains hashed session files and list/update semantics.
- Experiment retains IDs, CRUD/search APIs, and append behavior.
- Chat endpoints remain unchanged; persistence uses an injected canonical
  `PersistencePaths.sessions_root` and is independent of CWD.
- Universe remains one-object-per-file and keeps construction/query APIs; unsafe
  IDs are rejected instead of rewritten.
- Invalid records are not silently replaced. Listings retain valid records and
  expose structured bad-file diagnostics.

## Concerns

- Persisted response models add `schema_version` and `revision`; this is additive.
- No Phase 5 work was started.

## Reviewer remediation

The correction pass migrated the actual NiceGUI chat page from its CWD-relative
duplicate JSON store to `ChatSessionRepository`. Its legacy payload adapter keeps
the session ID, title, messages, counter and preview, and lazy migration remains
locked and idempotent. The UI shows a degraded warning for corrupt records and
the Chat API exposes storage status/count headers without changing its list body.

Todo, Experiment, Chat and Universe models/repositories now enforce literal schema
version 2, non-negative revisions, record-key identity, and optional CAS where
mutations replace mutable state. Universe save uses one locked upsert. Todo and
Experiment constructors used by tools, orchestration, CLI and workflow routes now
receive canonical locks/quarantine roots. Arbitrary Universe registry overrides
are rejected; a wired DataLake determines the canonical registry root.

Experiment and Universe list tools keep their existing result keys while adding
`status=DEGRADED` and structured diagnostics when corrupt records are skipped.

Additional verified seams include Todo and Chat stale-revision rejection and
legacy NiceGUI migration under a changed CWD. No new reviewer-specific test
produced a RED after implementation; this limitation is reported explicitly
rather than relabeling first-run GREEN tests as TDD RED evidence.

## Second reviewer remediation

NiceGUI now retains the complete canonical `ChatSession` and `ChatMessage`
objects while presenting UI-friendly wrappers. An unedited API-created session
round-trips through UI load/save without changing IDs, timestamps, context,
messages or revision. UI-specific counter/preview values merge only into
`context.legacy_ui`. Edited saves use the loaded revision as CAS and retain that
revision only after a successful write; stale saves raise a visible structured
conflict. Delete failures are surfaced and no longer suppressed.

Universe startup discovers the exact previous root `data/universes/registry`
under a migration lock, validates each record, atomically creates missing
canonical records under `data/registries/universes`, and leaves the source in
place. Repeated discovery is idempotent. Universe save has an explicit
same-ID last-locked-writer policy unless CAS is supplied.

Todo mutation tool schemas and Universe save schemas now advertise nonnegative
`expected_revision`. Todo results include schema/revision, and Universe
save/list/inspect results expose the persisted revision. Explicit experiment
roots remain explicit; only lock and quarantine infrastructure comes from the
canonical settings paths.

This correction added tests before the final schema placement correction, which
produced one genuine failure: CAS was accidentally declared on the validation
tool instead of the save tool. The lossless/UI/old-root tests were added after
their implementation and are therefore reported as regression evidence, not
TDD RED evidence.
