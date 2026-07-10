"""Canonical dependency resolution for file-backed repositories."""

from __future__ import annotations

from qmt_agent_trader.core.config import get_settings
from qmt_agent_trader.persistence.atomic_files import AtomicFileStore
from qmt_agent_trader.persistence.errors import StorageConflictError
from qmt_agent_trader.persistence.locks import LockManager
from qmt_agent_trader.persistence.paths import PersistencePaths


def resolve_file_repository_dependencies(
    *,
    lock_manager: LockManager | None,
    atomic_store: AtomicFileStore | None,
) -> tuple[LockManager, AtomicFileStore]:
    """Return one canonical lock namespace and its atomic file store.

    Root-only compatibility construction deliberately resolves through the same
    ``PersistencePaths.locks_root`` used by Web, Agent, workflow, and CLI entry
    points. This prevents an alternate registry-local lock namespace.
    """
    manager = lock_manager
    if manager is None and atomic_store is not None:
        manager = atomic_store.lock_manager
    if manager is None:
        settings = get_settings()
        paths = PersistencePaths.from_settings(settings)
        manager = LockManager(
            paths.locks_root,
            timeout_seconds=settings.remote_data_lock_timeout_seconds,
        )
    store = atomic_store or AtomicFileStore(manager)
    if store.lock_manager.locks_root != manager.locks_root:
        raise StorageConflictError(
            store_name="repositories",
            path=manager.locks_root,
            operation="resolve_dependencies",
            reason="atomic file store and repository use different lock namespaces",
        )
    return manager, store
