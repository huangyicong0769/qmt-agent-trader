from __future__ import annotations

from datetime import date, timedelta

import pandas as pd

from qmt_agent_trader.agent.permissions import ToolCapability
from qmt_agent_trader.agent.runtime import build_default_runtime
from qmt_agent_trader.agent.tool_registry import ToolDefinition, ToolRegistry
from qmt_agent_trader.core.config import Settings
from qmt_agent_trader.core.errors import PermissionDeniedError


def test_default_runtime_lists_and_summarizes_data_lake(tmp_path) -> None:
    runtime = build_default_runtime(
        Settings(
            project_root=tmp_path,
            qmt_gateway_api_key=None,
            qmt_gateway_hmac_secret=None,
            deepseek_api_key=None,
        )
    )
    runtime.lake.write_parquet(
        pd.DataFrame(
            [
                {
                    "ts_code": "000001.SZ",
                    "trade_date": "20240102",
                    "open": 10.0,
                    "high": 10.0,
                    "low": 9.5,
                    "close": 10.0,
                    "limit_up": True,
                },
                {
                    "ts_code": "000001.SZ",
                    "trade_date": "20240103",
                    "open": 9.0,
                    "high": 9.2,
                    "low": 9.0,
                    "close": 9.0,
                    "limit_down": True,
                },
            ]
        ),
        "raw",
        "tushare_daily_fixture",
    )

    datasets = runtime.call_tool("list_datasets", layer="raw")
    summary = runtime.call_tool("summarize_daily_bars")

    assert datasets["layers"]["raw"] == ["tushare_daily_fixture"]
    assert summary["rows"] == 2
    assert summary["trade_state_counts"]["limit_up"] == 1
    assert summary["trade_state_counts"]["limit_down"] == 1


def test_default_runtime_can_compute_factor_tool(tmp_path) -> None:
    runtime = build_default_runtime(
        Settings(
            project_root=tmp_path,
            qmt_gateway_api_key=None,
            qmt_gateway_hmac_secret=None,
            deepseek_api_key=None,
        )
    )
    start = date(2024, 1, 1)
    rows = [
        {
            "ts_code": "000001.SZ",
            "trade_date": f"{start + timedelta(days=offset):%Y%m%d}",
            "open": 10.0 + offset,
            "high": 11.0 + offset,
            "low": 9.0 + offset,
            "close": 10.0 + offset,
        }
        for offset in range(21)
    ]
    runtime.lake.write_parquet(pd.DataFrame(rows), "raw", "tushare_daily_fixture")

    result = runtime.call_tool("compute_factor", name="momentum_20d", date="20240121")

    assert result["status"] == "computed"
    assert result["non_null"] == 1


def test_default_runtime_plans_sensitivity_analysis(tmp_path) -> None:
    runtime = build_default_runtime(
        Settings(
            project_root=tmp_path,
            qmt_gateway_api_key=None,
            qmt_gateway_hmac_secret=None,
            deepseek_api_key=None,
        )
    )

    result = runtime.call_tool(
        "plan_sensitivity_analysis",
        cost_multipliers=[1.0, 2.0],
        slippage_bps=[0.0],
        execution_delay_days=[1, 2],
        top_n=[10],
        max_single_position_pct=[0.1],
    )

    assert result["status"] == "planned"
    assert result["scenario_count"] == 4
    assert result["runner_contract"]["required_metrics"] == [
        "total_return",
        "sharpe",
        "max_drawdown",
        "turnover",
        "diagnostic_pass",
    ]


def test_default_runtime_runs_factor_rank_sensitivity(tmp_path) -> None:
    runtime = build_default_runtime(
        Settings(
            project_root=tmp_path,
            qmt_gateway_api_key=None,
            qmt_gateway_hmac_secret=None,
            deepseek_api_key=None,
        )
    )
    start = date(2024, 1, 1)
    rows = []
    for offset in range(24):
        trade_date = f"{start + timedelta(days=offset):%Y%m%d}"
        rows.append(
            {
                "ts_code": "000001.SZ",
                "trade_date": trade_date,
                "open": 10.0 + offset,
                "high": 11.0 + offset,
                "low": 9.0 + offset,
                "close": 10.0 + offset,
            }
        )
        rows.append(
            {
                "ts_code": "000002.SZ",
                "trade_date": trade_date,
                "open": 20.0 + offset * 0.1,
                "high": 21.0 + offset * 0.1,
                "low": 19.0 + offset * 0.1,
                "close": 20.0 + offset * 0.1,
            }
        )
    runtime.lake.write_parquet(pd.DataFrame(rows), "raw", "tushare_daily_fixture")

    result = runtime.call_tool(
        "run_factor_rank_sensitivity",
        factor_name="momentum_20d",
        cost_multipliers=[1.0, 2.0],
        slippage_bps=[0.0],
        execution_delay_days=[1],
        top_n=[1],
        max_single_position_pct=[0.5],
        initial_cash=100000,
    )

    assert result["status"] == "completed"
    assert result["summary"]["scenario_count"] == 2
    assert result["summary"]["pass_ratio"] == 1.0


def test_default_runtime_persists_factor_rank_research_report(tmp_path) -> None:
    runtime = build_default_runtime(
        Settings(
            project_root=tmp_path,
            qmt_gateway_api_key=None,
            qmt_gateway_hmac_secret=None,
            deepseek_api_key=None,
        )
    )
    start = date(2024, 1, 1)
    rows = []
    for offset in range(24):
        trade_date = f"{start + timedelta(days=offset):%Y%m%d}"
        rows.append(
            {
                "ts_code": "000001.SZ",
                "trade_date": trade_date,
                "open": 10.0 + offset,
                "high": 11.0 + offset,
                "low": 9.0 + offset,
                "close": 10.0 + offset,
            }
        )
        rows.append(
            {
                "ts_code": "000002.SZ",
                "trade_date": trade_date,
                "open": 20.0 + offset * 0.1,
                "high": 21.0 + offset * 0.1,
                "low": 19.0 + offset * 0.1,
                "close": 20.0 + offset * 0.1,
            }
        )
    runtime.lake.write_parquet(pd.DataFrame(rows), "raw", "tushare_daily_fixture")

    receipt = runtime.call_tool(
        "run_factor_rank_sensitivity_report",
        factor_name="momentum_20d",
        cost_multipliers=[1.0],
        slippage_bps=[0.0],
        execution_delay_days=[1],
        top_n=[1],
        max_single_position_pct=[0.5],
        initial_cash=100000,
        agent_notes="candidate passed the smoke robustness grid",
        infrastructure_requests=["add capacity stress checks"],
    )
    compared = runtime.call_tool("compare_research_reports", limit=5)

    assert receipt["status"] == "saved"
    assert receipt["research_only"] is True
    assert receipt["live_trading_allowed"] is False
    assert compared["status"] == "compared"
    assert compared["runs"][0]["summary"]["scenario_count"] == 1
    assert compared["infrastructure_requests"] == ["add capacity stress checks"]


def test_registry_deepseek_tools_keep_permission_guard() -> None:
    registry = ToolRegistry()
    registry.register(
        ToolDefinition(
            name="submit_order",
            capability=ToolCapability.SUBMIT_ORDER,
            fn=lambda: {"status": "should_not_run"},
        )
    )

    try:
        registry.deepseek_tools_for_llm()
    except PermissionDeniedError as exc:
        assert "SUBMIT_ORDER" in str(exc)
    else:
        raise AssertionError("submit order tool should be denied to LLM")
