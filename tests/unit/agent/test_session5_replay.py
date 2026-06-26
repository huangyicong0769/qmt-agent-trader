from __future__ import annotations

from datetime import date, timedelta

import pandas as pd

from qmt_agent_trader.agent.sandbox import CodeSandbox
from qmt_agent_trader.agent.schemas import ToolContext
from qmt_agent_trader.agent.tools import build_agent_registry
from qmt_agent_trader.data.storage import DataLake


def _seed_etf_bars(lake: DataLake, *, symbols: tuple[str, ...] = ("159259.SZ",)) -> None:
    start = date(2025, 8, 28)
    rows = []
    for offset in range(90):
        trade_date = f"{start + timedelta(days=offset):%Y%m%d}"
        for index, symbol in enumerate(symbols):
            drift = offset * (0.01 if symbol == "159259.SZ" else 0.002)
            base = 1.0 + index * 0.5
            rows.append(
                {
                    "ts_code": symbol,
                    "trade_date": trade_date,
                    "open": base + drift,
                    "high": base + drift + 0.03,
                    "low": base + drift - 0.02,
                    "close": base + drift + (0.01 if offset % 5 else -0.01),
                    "vol": 1_000_000 + offset * 1000,
                    "amount": 1_000_000 + offset * 2000,
                }
            )
    lake.write_parquet(pd.DataFrame(rows), "raw", "tushare_fund_daily")


def _registry(tmp_path, lake: DataLake):
    return build_agent_registry(
        data_lake=lake,
        audit_path=tmp_path / "audit.jsonl",
        experiment_root=tmp_path / "experiments",
        sandbox=CodeSandbox(tmp_path / "generated"),
    )


def test_session5_factor_spec_direct_args_are_preserved(tmp_path) -> None:
    lake = DataLake(root=tmp_path / "lake", duckdb_path=tmp_path / "db.duckdb")
    registry = _registry(tmp_path, lake)

    result = registry.run_tool(
        "create_factor_spec",
        {
            "factor_name": "momentum_20d",
            "factor_description": "20日动量因子：过去20个交易日的累计收益率。",
            "universe": "159259.SZ",
            "data_sources": ["tushare_fund_daily"],
        },
        ToolContext(run_id="s5"),
    )

    spec = result["factor_spec"]
    assert spec["name"] == "momentum_20d"
    assert spec["inputs"] == ["tushare_fund_daily"]
    assert "20日动量" in spec["formula"]


def test_session5_empty_factor_tool_args_return_structured_errors(tmp_path) -> None:
    lake = DataLake(root=tmp_path / "lake", duckdb_path=tmp_path / "db.duckdb")
    registry = _registry(tmp_path, lake)
    context = ToolContext(run_id="s5", experiment_id="exp_s5")

    generated = registry.run_tool("generate_factor_code", {}, context)
    static = registry.run_tool("run_factor_static_checks", {}, context)
    evaluated = registry.run_tool("evaluate_factor_candidate", {}, context)

    assert generated["status"] == "INVALID_REQUEST"
    assert static["status"] == "INVALID_REQUEST"
    assert evaluated["status"] == "INVALID_REQUEST"


def test_session5_factor_tool_descriptions_expose_required_inputs(tmp_path) -> None:
    lake = DataLake(root=tmp_path / "lake", duckdb_path=tmp_path / "db.duckdb")
    registry = _registry(tmp_path, lake)
    context = ToolContext(run_id="s5")

    expected_required = {
        "generate_factor_code": ["factor_spec"],
        "run_factor_static_checks": ["code_path"],
        "save_factor": ["factor_id", "code_path"],
        "evaluate_factor_candidate": ["factor_id"],
        "run_backtest": ["factor_name"],
    }
    for tool_name, required in expected_required.items():
        described = registry.run_tool("describe_tool", {"name": tool_name}, context)
        schema = described["tool_spec"]["input_schema"]
        assert schema["type"] == "object"
        for field in required:
            assert field in schema["properties"]
            assert field in schema.get("required", [])


def test_session5_single_etf_factor_evaluation_uses_time_series_mode(tmp_path) -> None:
    lake = DataLake(root=tmp_path / "lake", duckdb_path=tmp_path / "db.duckdb")
    _seed_etf_bars(lake)
    registry = _registry(tmp_path, lake)

    result = registry.run_tool(
        "evaluate_factor_candidate",
        {
            "factor_id": "momentum_20d",
            "symbol": "159259.SZ",
            "start_date": "2025-09-30",
            "end_date": "2025-11-20",
        },
        ToolContext(run_id="s5"),
    )

    assert result["status"] == "validated"
    assert result["evaluation_mode"] == "time_series"
    assert result["symbols"] == ["159259.SZ"]
    assert result["time_series"]["observations"] > 0
    assert result["time_series"]["spearman_ic"] is not None

    cached = registry.run_tool(
        "evaluate_factor_candidate",
        {
            "factor_id": "momentum_20d",
            "symbol": "159259.SZ",
            "start_date": "2025-09-30",
            "end_date": "2025-11-20",
        },
        ToolContext(run_id="s5-repeat"),
    )
    assert cached["status"] == "validated"
    assert cached["cache_hit"] is True


def test_session5_run_backtest_can_scope_to_single_etf(tmp_path) -> None:
    lake = DataLake(root=tmp_path / "lake", duckdb_path=tmp_path / "db.duckdb")
    _seed_etf_bars(lake, symbols=("159259.SZ", "000001.SZ", "000002.SZ"))
    registry = _registry(tmp_path, lake)

    result = registry.run_tool(
        "run_backtest",
        {
            "factor_name": "momentum_20d",
            "symbol": "159259.SZ",
            "start_date": "2025-09-30",
            "end_date": "2025-11-20",
            "top_n": 1,
        },
        ToolContext(run_id="s5"),
    )

    assert result["status"] == "completed"
    assert result["symbols"] == ["159259.SZ"]
    assert result["metrics"]["trade_count"] > 0
