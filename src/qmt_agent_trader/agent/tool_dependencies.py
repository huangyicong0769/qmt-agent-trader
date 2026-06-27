"""Runtime-scoped dependencies for Agent tools."""

from __future__ import annotations

from dataclasses import dataclass

from qmt_agent_trader.agent.audit import AuditLogger
from qmt_agent_trader.agent.experiment_store import ExperimentStore
from qmt_agent_trader.agent.sandbox import CodeSandbox
from qmt_agent_trader.core.config import Settings
from qmt_agent_trader.data.storage import DataLake


@dataclass(frozen=True)
class AgentToolDependencies:
    settings: Settings
    data_lake: DataLake
    sandbox: CodeSandbox
    experiment_store: ExperimentStore
    audit_logger: AuditLogger
