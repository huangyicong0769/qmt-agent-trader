# Preserve-Raw Storage Reset Design

The storage CLI needs an explicit destructive reset for development periods where legacy control
state and Agent artifacts are incompatible with the current persistence contracts. The reset
preserves only validated raw provider Parquet and repository configuration/source.

Planning is read-only and deterministic. It inventories governed reset roots, rejects symbolic
links and corrupt raw data, hashes every deleted and preserved file, and returns a confirmation
digest. Execution recomputes the plan under the storage maintenance barrier, atomically stages
targets, initializes current migrations, verifies raw identity and full storage health, then
permanently removes staging. Failures restore the staged state. A successful receipt records only
counts, hashes, timestamps, profile, and verification outcome.

The implementation reuses `PersistencePaths`, `LockManager`, `AtomicFileStore`, migration registry,
and `StorageOperations`. It introduces no CWD-derived roots, compatibility adoption, or silent
repair. The only supported profile is `preserve-raw`.
