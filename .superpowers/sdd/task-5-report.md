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
