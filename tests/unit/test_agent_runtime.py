from __future__ import annotations

from pathlib import Path

import pandas as pd

from qmt_agent_trader.agent.permissions import PermissionLevel, ToolCallMode
from qmt_agent_trader.agent.runtime import _default_research_system_prompt, build_default_runtime
from qmt_agent_trader.agent.schemas import ToolContext
from qmt_agent_trader.core.config import Settings
from qmt_agent_trader.core.errors import PermissionDeniedError


def _runtime(tmp_path):
    return build_default_runtime(
        Settings(
            project_root=tmp_path,
            qmt_gateway_api_key=None,
            qmt_gateway_hmac_secret=None,
            deepseek_api_key=None,
        )
    )


def test_runtime_exposes_one_agent_callable_surface_for_web_and_llm(tmp_path) -> None:
    runtime = _runtime(tmp_path)

    agent_tool_names = {
        item["name"] for item in runtime.list_tools(agent_callable_only=True)
    }
    llm_tool_names = {item.name for item in runtime.llm_tools(run_id="run-unified")}

    assert agent_tool_names == llm_tool_names
    assert "query_bars" in agent_tool_names
    assert "list_tushare_capabilities" in agent_tool_names
    assert "plan_tushare_fetch" in agent_tool_names
    assert "run_tushare_fetch" in agent_tool_names
    assert "build_data_table" in agent_tool_names
    assert "run_remote_data_update" not in agent_tool_names
    assert "run_shell_command" not in agent_tool_names
    assert "propose_tool_registration" not in agent_tool_names


def test_default_runtime_generates_under_injected_project_root(tmp_path) -> None:
    runtime = _runtime(tmp_path)
    result = runtime.run_tool(
        "generate_factor_code",
        {
            "factor_spec": {
                "factor_id": "factor_settings_root",
                "name": "settings root",
                "formula": "close",
            },
            "python_function": (
                "def compute_factor(data: pd.DataFrame, context: FactorContext) -> pd.Series:\n"
                "    return pd.Series(data['close'], index=data.index, name='factor_value')"
            ),
        },
        ToolContext(run_id="settings-run"),
    )

    expected = tmp_path / "src/qmt_agent_trader/agent/generated"
    assert Path(result["code_path"]).is_relative_to(expected)


def test_default_prompt_keeps_local_quant_workflows_on_native_tools() -> None:
    prompt = _default_research_system_prompt()

    assert "prefer native data, factor, backtest, and report tools" in prompt
    assert "do not call external MCP/web tools unless" in prompt
    assert "Do not blame replay, validation, or test protocols" in prompt
    assert "observed tool status" in prompt
    assert "preserves your raw final answer" in prompt
    assert "reports evidence-output conflicts" in prompt
    assert "unless a tool returned NOT_CONFIGURED" in prompt


def test_runtime_instances_keep_tool_dependencies_isolated(tmp_path) -> None:
    runtime_a = _runtime(tmp_path / "a")
    runtime_b = _runtime(tmp_path / "b")
    runtime_a.lake.write_parquet(
        pd.DataFrame(
            [
                {
                    "ts_code": "000001.SZ",
                    "trade_date": "20260105",
                    "open": 10.0,
                    "high": 10.5,
                    "low": 9.8,
                    "close": 10.2,
                    "vol": 100,
                }
            ]
        ),
        "raw",
        "tushare/daily",
    )
    runtime_b.lake.write_parquet(
        pd.DataFrame(
            [
                {
                    "ts_code": "000002.SZ",
                    "trade_date": "20260105",
                    "open": 20.0,
                    "high": 20.5,
                    "low": 19.8,
                    "close": 20.2,
                    "vol": 200,
                }
            ]
        ),
        "raw",
        "tushare/daily",
    )

    result_a = runtime_a.run_tool(
        "query_bars",
        {
            "symbol": "000001.SZ",
            "start_date": "20260101",
            "end_date": "20260110",
        },
        ToolContext(run_id="runtime-a", call_mode=ToolCallMode.AUTONOMOUS_AGENT),
    )

    assert result_a["metadata"]["returned"] == 1
    assert result_a["rows"][0]["symbol"] == "000001.SZ"


def test_runtime_permission_modes_allow_review_tools_only_for_internal_workflows(
    tmp_path,
) -> None:
    runtime = _runtime(tmp_path)

    spec = runtime.describe_tool("propose_tool_registration")
    assert spec.permission == PermissionLevel.APPROVAL_REQUIRED
    assert "propose_tool_registration" not in {
        item.name for item in runtime.llm_tools(run_id="run-permissions")
    }

    try:
        runtime.run_tool(
            "propose_tool_registration",
            {"tool_candidate_id": "candidate", "score": {"overall": 0.9}},
            ToolContext(
                run_id="autonomous-denied",
                call_mode=ToolCallMode.AUTONOMOUS_AGENT,
            ),
        )
    except PermissionDeniedError:
        pass
    else:
        raise AssertionError("autonomous agent must not run approval-required tools")
