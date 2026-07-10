"""Canonical, injectable persistence paths."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from qmt_agent_trader.core.config import Settings


@dataclass(frozen=True)
class PersistencePaths:
    project_root: Path
    data_root: Path
    lake_root: Path
    control_db_path: Path
    artifact_root: Path
    reports_root: Path
    approvals_root: Path
    order_plans_root: Path
    sessions_root: Path
    experiments_root: Path
    registries_root: Path
    cache_root: Path
    audit_root: Path
    locks_root: Path
    quarantine_root: Path
    backup_root: Path

    @classmethod
    def from_settings(cls, settings: Settings) -> PersistencePaths:
        project = settings.project_root.expanduser().resolve()
        data = _under(project, settings.data_dir)
        logs = _under(project, settings.log_dir)
        artifacts = (project / "artifacts").resolve()
        return cls(
            project_root=project,
            data_root=data,
            lake_root=(data / "lake").resolve(),
            control_db_path=(data / "qmt_agent_trader.duckdb").resolve(),
            artifact_root=artifacts,
            reports_root=(project / "reports").resolve(),
            approvals_root=(project / "approvals").resolve(),
            order_plans_root=(project / "order_plans").resolve(),
            sessions_root=(project / "sessions").resolve(),
            experiments_root=(data / "experiments").resolve(),
            registries_root=(data / "registries").resolve(),
            cache_root=(project / "reports/cache").resolve(),
            audit_root=(logs / "audit").resolve(),
            locks_root=(data / "locks").resolve(),
            quarantine_root=(data / "quarantine").resolve(),
            backup_root=(data / "backups").resolve(),
        )


def _under(project: Path, configured: Path) -> Path:
    return (configured if configured.is_absolute() else project / configured).expanduser().resolve()
