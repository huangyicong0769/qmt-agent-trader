from __future__ import annotations

from qmt_agent_trader.agent.audit import AuditLogger
from qmt_agent_trader.agent.experiment_store import ExperimentStore
from qmt_agent_trader.agent.sandbox import CodeSandbox
from qmt_agent_trader.agent.schemas import ToolContext
from qmt_agent_trader.agent.tool_dependencies import AgentToolDependencies
from qmt_agent_trader.agent.tool_result import ExecutionStatus, normalize_tool_result
from qmt_agent_trader.agent.tools.remote_data_tools import build_remote_data_tools
from qmt_agent_trader.core.config import Settings
from qmt_agent_trader.data.storage import DataLake


def test_registry_driven_data_tools_return_structured_statuses(tmp_path) -> None:
    lake = DataLake(root=tmp_path / "lake", duckdb_path=tmp_path / "db.duckdb")
    deps = AgentToolDependencies(
        settings=Settings(project_root=tmp_path, tushare_token=None),
        data_lake=lake,
        sandbox=CodeSandbox(tmp_path / "generated"),
        experiment_store=ExperimentStore(tmp_path / "experiments"),
        audit_logger=AuditLogger(tmp_path / "audit.jsonl"),
    )
    tools = {tool.spec.name: tool for tool in build_remote_data_tools(deps)}
    payloads = {
        "plan_tushare_fetch": tools["plan_tushare_fetch"].run(
            {
                "items": [
                    {
                        "api_name": "daily_basic",
                        "symbols": ["000001.SZ"],
                        "fields": ["ts_code", "trade_date", "pe_ttm"],
                        "start_date": "20260708",
                        "end_date": "20260708",
                    }
                ]
            },
            ToolContext(run_id="structured-plan"),
        ),
        "run_tushare_fetch": tools["run_tushare_fetch"].run(
            {
                "items": [
                    {
                        "api_name": "daily_basic",
                        "symbols": ["000001.SZ"],
                        "fields": ["ts_code", "trade_date", "pe_ttm"],
                        "start_date": "20260708",
                        "end_date": "20260708",
                    }
                ],
                "dry_run": True,
            },
            ToolContext(run_id="structured-run"),
        ),
        "build_data_table": tools["build_data_table"].run(
            {"table": "security_master"},
            ToolContext(run_id="structured-build"),
        ),
    }

    for tool_name, payload in payloads.items():
        normalized = normalize_tool_result(
            tool_name,
            payload,
            execution_status=ExecutionStatus.OK,
        )
        assert normalized["domain_status"] != "UNKNOWN"
        assert normalized["evidence_status"] != "UNKNOWN"
        assert "legacy_unstructured_tool_result" not in normalized["warnings"]
