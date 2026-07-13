"""Declarative catalog of logical authoritative local stores."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from qmt_agent_trader.persistence.paths import PersistencePaths

StoreKind = Literal["duckdb", "parquet", "json", "jsonl", "artifact", "code"]
StoreLayout = Literal["single_file", "directory"]


@dataclass(frozen=True)
class StoreDefinition:
    name: str
    kind: StoreKind
    path: Path
    owner: str
    source_of_truth: str
    schema_version: int | None
    mutable: bool
    lock_resource: str
    backup: str
    layout: StoreLayout
    governed: bool = False
    verifier_id: str = "generic"


@dataclass(frozen=True)
class StoreCatalog:
    stores: tuple[StoreDefinition, ...]

    @classmethod
    def canonical(cls, paths: PersistencePaths) -> StoreCatalog:
        data = paths.data_root
        definitions = (
            StoreDefinition(
                "control_db",
                "duckdb",
                paths.control_db_path,
                "DataLake",
                "catalog/control-state",
                1,
                True,
                str(paths.control_db_path),
                "checkpoint-copy",
                "single_file",
            ),
            StoreDefinition(
                "lake_raw",
                "parquet",
                paths.lake_root / "raw",
                "DataLake",
                "authoritative-provider-bytes",
                None,
                True,
                str(paths.lake_root / "raw"),
                "copy",
                "directory",
            ),
            StoreDefinition(
                "lake_silver",
                "parquet",
                paths.lake_root / "silver",
                "DataLake",
                "derived-canonical",
                None,
                True,
                str(paths.lake_root / "silver"),
                "copy",
                "directory",
            ),
            StoreDefinition(
                "lake_gold",
                "parquet",
                paths.lake_root / "gold",
                "DataLake",
                "derived-canonical",
                None,
                True,
                str(paths.lake_root / "gold"),
                "copy",
                "directory",
            ),
            StoreDefinition(
                "lake_metadata",
                "parquet",
                paths.lake_root / "metadata",
                "DataLake",
                "migration-metadata",
                None,
                True,
                str(paths.lake_root / "metadata"),
                "copy",
                "directory",
            ),
            StoreDefinition(
                "factor_registry",
                "json",
                data / "factors/registry.json",
                "FactorRegistry",
                "authoritative",
                2,
                True,
                str(data / "factors/registry.json"),
                "copy",
                "single_file",
                verifier_id="versioned_registry_v2",
            ),
            StoreDefinition(
                "strategy_registry",
                "json",
                data / "strategies/registry.json",
                "StrategyRegistry",
                "authoritative",
                2,
                True,
                str(data / "strategies/registry.json"),
                "copy",
                "single_file",
                verifier_id="versioned_registry_v2",
            ),
            StoreDefinition(
                "todos",
                "json",
                data / "todos",
                "TodoListStore",
                "authoritative",
                2,
                True,
                str(data / "todos"),
                "copy",
                "directory",
                verifier_id="versioned_record_todos_v2",
            ),
            StoreDefinition(
                "experiments",
                "json",
                paths.experiments_root,
                "ExperimentStore",
                "authoritative",
                2,
                True,
                str(paths.experiments_root),
                "copy",
                "directory",
                verifier_id="versioned_record_experiments_v2",
            ),
            StoreDefinition(
                "sessions",
                "json",
                paths.sessions_root,
                "ChatSessionRepository",
                "authoritative",
                2,
                True,
                str(paths.sessions_root),
                "copy",
                "directory",
                verifier_id="versioned_record_sessions_v2",
            ),
            StoreDefinition(
                "universes",
                "json",
                paths.registries_root / "universes",
                "UniverseRegistry",
                "authoritative",
                2,
                True,
                str(paths.registries_root / "universes"),
                "copy",
                "directory",
                verifier_id="versioned_record_universes_v2",
            ),
            StoreDefinition(
                "approvals",
                "artifact",
                paths.approvals_root,
                "StrategyApproval",
                "governance",
                1,
                False,
                f"artifact-store:{paths.approvals_root.resolve()}",
                "copy",
                "directory",
                True,
            ),
            StoreDefinition(
                "order_plans",
                "artifact",
                paths.order_plans_root,
                "OrderPlanService",
                "governance+events",
                1,
                True,
                f"artifact-store:{paths.order_plans_root.resolve()}",
                "copy",
                "directory",
                True,
            ),
            StoreDefinition(
                "order_plan_events",
                "jsonl",
                paths.order_plans_root / ".events",
                "OrderPlanService",
                "governance-events",
                1,
                True,
                f"artifact-store:{paths.order_plans_root.resolve()}",
                "copy",
                "directory",
                verifier_id="order_plan_event_stream_v1",
            ),
            StoreDefinition(
                "backtest_reports",
                "artifact",
                paths.reports_root / "backtests",
                "BacktestService",
                "backtest-evidence",
                1,
                False,
                f"artifact-store:{(paths.reports_root / 'backtests').resolve()}",
                "copy",
                "directory",
                True,
            ),
            StoreDefinition(
                "research_reports",
                "artifact",
                paths.reports_root / "research",
                "ResearchReportService",
                "research-evidence",
                1,
                False,
                f"artifact-store:{(paths.reports_root / 'research').resolve()}",
                "copy",
                "directory",
                True,
            ),
            StoreDefinition(
                "audit",
                "jsonl",
                paths.audit_root,
                "AuditJsonlStore",
                "audit-source",
                1,
                True,
                str(paths.audit_root),
                "copy",
                "directory",
            ),
            StoreDefinition(
                "generated_code",
                "code",
                paths.project_root / "src/qmt_agent_trader/agent/generated",
                "CodeSandbox",
                "review-candidates",
                1,
                False,
                "artifact-store:"
                + str(
                    (paths.project_root / "src/qmt_agent_trader/agent/generated").resolve()
                ),
                "copy",
                "directory",
                True,
            ),
        )
        return cls(definitions)

    def by_name(self, name: str) -> StoreDefinition:
        for store in self.stores:
            if store.name == name:
                return store
        raise KeyError(name)
