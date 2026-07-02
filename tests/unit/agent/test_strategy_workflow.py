"""Tests for the Strategy Engineering workflow."""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import pytest

from qmt_agent_trader.agent.experiment_store import ExperimentStore
from qmt_agent_trader.agent.sandbox import CodeSandbox
from qmt_agent_trader.agent.schemas import ExperimentStatus, ToolContext
from qmt_agent_trader.agent.tools import build_agent_registry
from qmt_agent_trader.agent.workflows.strategy_engineering import (
    StrategyEngineeringWorkflow,
)
from qmt_agent_trader.data.storage import DataLake


@pytest.fixture
def lake(tmp_path):
    return DataLake(
        root=tmp_path / "lake",
        duckdb_path=tmp_path / "test.duckdb",
    )


@pytest.fixture
def registry(lake, tmp_path):
    reg = build_agent_registry(
        data_lake=lake,
        audit_path=tmp_path / "audit.jsonl",
        experiment_root=tmp_path / "experiments",
        sandbox=CodeSandbox(tmp_path / "generated"),
    )
    import pandas as pd

    lake.write_parquet(
        pd.DataFrame(
            {
                "ts_code": ["000001.SZ"],
                "trade_date": ["20240102"],
                "open": [10],
                "high": [11],
                "low": [9],
                "close": [10.5],
                "vol": [1000000],
                "amount": [10000000],
            }
        ),
        "raw",
        "tushare/daily",
    )
    return reg


@pytest.fixture
def store(tmp_path):
    return ExperimentStore(tmp_path / "experiments")


def test_strategy_workflow_runs(registry, store):
    wf = StrategyEngineeringWorkflow(registry, store)
    exp = wf.run(
        "基于动量的日频轮动策略",
        ["momentum_20d"],
        "stock_etf",
        "20200101",
        "20240624",
    )
    assert exp.experiment_id is not None
    assert exp.kind == "strategy_engineering"


def test_strategy_workflow_creates_experiment(registry, store):
    wf = StrategyEngineeringWorkflow(registry, store)
    exp = wf.run("test", ["reversal_5d"], "stock", "20200101", "20240624")
    assert exp.status in (ExperimentStatus.REVIEW_REQUIRED, ExperimentStatus.FAILED)


def test_strategy_workflow_does_not_call_broker(registry, store):
    """Ensure the strategy workflow cannot generate live broker interactions."""
    wf = StrategyEngineeringWorkflow(registry, store)
    exp = wf.run("safe test", ["momentum_20d"], "stock", "20200101", "20240624")
    # Verify the experiment artifacts contain no broker references
    artifacts = [a.lower() for a in exp.artifacts]
    assert not any("broker" in a for a in artifacts)
    assert not any("gateway" in a for a in artifacts)
    assert not any("submit_order" in a for a in artifacts)


def test_agent_can_list_generated_strategy_candidates(registry):
    context = ToolContext(run_id="strategy-list")
    spec = registry.run_tool(
        "create_strategy_spec",
        {
            "strategy_idea": "基于动量的候选策略",
            "selected_factors": ["momentum_20d"],
        },
        context,
    )["strategy_spec"]
    generated = registry.run_tool("generate_strategy_code", {"strategy_spec": spec}, context)

    listed = registry.run_tool(
        "list_strategy_candidates",
        {"query": spec["strategy_id"]},
        context,
    )

    assert generated["status"] == "generated"
    assert listed["status"] == "ok"
    assert listed["count"] == 1
    assert listed["strategies"][0]["strategy_id"] == spec["strategy_id"]
    assert listed["strategies"][0]["status"] == "draft"


def test_agent_can_save_generated_strategy_candidate(registry):
    context = ToolContext(run_id="strategy-save")
    spec = registry.run_tool(
        "create_strategy_spec",
        {
            "strategy_idea": "基于动量和波动率的候选策略",
            "selected_factors": ["momentum_20d", "volatility_20d"],
        },
        context,
    )["strategy_spec"]
    generated = registry.run_tool("generate_strategy_code", {"strategy_spec": spec}, context)

    saved = registry.run_tool(
        "save_strategy_candidate",
        {
            "strategy_spec": spec,
            "code_path": generated["code_path"],
            "tests_path": generated["tests_path"],
        },
        context,
    )
    listed = registry.run_tool(
        "list_strategy_candidates",
        {"query": spec["strategy_id"]},
        context,
    )

    assert saved["status"] == "saved"
    assert saved["saved_strategy"]["status"] == "GENERATED_BY_LLM"
    assert saved["live_trading_allowed"] is False
    assert listed["strategies"][0]["saved_in_registry"] is True
    assert listed["strategies"][0]["status"] == "GENERATED_BY_LLM"


def test_saved_strategy_id_can_be_used_for_backtest(registry):
    context = ToolContext(run_id="strategy-save-backtest")
    spec = registry.run_tool(
        "create_strategy_spec",
        {
            "strategy_idea": "基于动量的候选策略",
            "selected_factors": ["momentum_20d"],
        },
        context,
    )["strategy_spec"]
    generated = registry.run_tool("generate_strategy_code", {"strategy_spec": spec}, context)
    registry.run_tool(
        "save_strategy_candidate",
        {
            "strategy_spec": spec,
            "code_path": generated["code_path"],
            "tests_path": generated["tests_path"],
        },
        context,
    )

    result = registry.run_tool(
        "run_backtest",
        {
            "strategy_id": spec["strategy_id"],
            "start_date": "20240102",
            "end_date": "20240102",
            "symbols": ["000001.SZ"],
            "top_n": 1,
        },
        context,
    )

    assert result["status"] != "error"
    assert result["strategy_id"] == spec["strategy_id"]


def test_multi_factor_strategy_backtest_executes_all_requested_factors(registry, lake):
    _write_multi_factor_bars(lake)
    context = ToolContext(run_id="strategy-multi-factor-backtest")
    spec = registry.run_tool(
        "create_strategy_spec",
        {
            "strategy_idea": "低波动和低换手 smart beta 组合",
            "selected_factors": ["volatility_20d", "turnover_20d"],
            "universe": "custom_cyclical_basket",
            "constraints": {"max_single_position_pct": 0.5},
        },
        context,
    )["strategy_spec"]

    result = registry.run_tool(
        "run_backtest",
        {
            "strategy_spec": spec,
            "start_date": "20240101",
            "end_date": "20240215",
            "symbols": ["000001.SZ", "000002.SZ", "000003.SZ"],
            "top_n": 1,
        },
        context,
    )

    assert result["status"] == "completed"
    assert result["requested_factor_ids"] == ["volatility_20d", "turnover_20d"]
    assert result["factor_ids"] == ["volatility_20d", "turnover_20d"]
    assert result["adapter_limitations"] == []
    assert result["execution_backend"] == "factor_rank_composite_adapter"
    diagnostic_checks = {
        check["name"]: check for check in result["diagnostics"]["checks"]
    }
    assert diagnostic_checks["positive_ic_ratio"]["status"] in {"PASS", "WARN"}
    assert diagnostic_checks["positive_ic_ratio"]["evidence_source"] == "computed"
    assert diagnostic_checks["positive_ic_ratio"]["observed"] >= 0
    assert diagnostic_checks["walk_forward_consistency"]["status"] in {"PASS", "WARN"}
    assert diagnostic_checks["walk_forward_consistency"]["evidence_source"] == "computed"


def test_ad_hoc_multi_factor_strategy_spec_without_id_is_backtestable(registry, lake):
    _write_multi_factor_bars(lake)
    context = ToolContext(run_id="strategy-ad-hoc-multi-factor")

    result = registry.run_tool(
        "run_backtest",
        {
            "strategy_spec": {
                "name": "临时低波低换手组合",
                "factors": [
                    {"factor_id": "volatility_20d", "weight": 0.7},
                    {"factor_id": "turnover_20d", "weight": 0.3},
                ],
                "portfolio": {"method": "equal_weight_top_n", "top_n": 1},
                "universe": "custom_cyclical_basket",
            },
            "start_date": "20240101",
            "end_date": "20240215",
            "symbols": ["000001.SZ", "000002.SZ", "000003.SZ"],
            "top_n": 1,
        },
        context,
    )

    assert result["status"] == "completed"
    assert result["strategy_id"].startswith("strat_")
    assert result["requested_factor_ids"] == ["volatility_20d", "turnover_20d"]
    assert result["factor_ids"] == ["volatility_20d", "turnover_20d"]
    assert result["execution_backend"] == "factor_rank_composite_adapter"


def test_run_backtest_resolves_cyclical_universe_when_symbols_are_omitted(registry, lake):
    _write_multi_factor_bars(lake)
    lake.write_parquet(
        pd.DataFrame(
            {
                "ts_code": ["000001.SZ", "000002.SZ", "000003.SZ"],
                "name": ["平安银行", "万科A", "贵州茅台"],
                "industry": ["银行", "房地产", "白酒"],
                "list_status": ["L", "L", "L"],
                "list_date": ["19910403", "19910129", "20010827"],
            }
        ),
        "raw",
        "tushare/stock_basic",
    )
    context = ToolContext(run_id="strategy-cyclical-resolved")

    result = registry.run_tool(
        "run_backtest",
        {
            "strategy_spec": {
                "name": "顺周期低波低换手组合",
                "factors": [
                    {"factor_id": "volatility_20d", "weight": 0.6},
                    {"factor_id": "turnover_20d", "weight": 0.4},
                ],
                "portfolio": {"method": "equal_weight_top_n", "top_n": 1},
                "universe": "cyclical",
            },
            "start_date": "20240101",
            "end_date": "20240215",
            "top_n": 1,
        },
        context,
    )

    assert result["status"] == "completed"
    assert result["symbols"] == ["000001.SZ", "000002.SZ"]
    assert result["universe_resolution"]["metadata"]["theme"] == "cyclical"
    assert result["execution_backend"] == "factor_rank_composite_adapter"


def test_empty_strategy_spec_returns_invalid_request(registry):
    context = ToolContext(run_id="strategy-empty-spec")

    result = registry.run_tool("run_backtest", {"strategy_spec": {}}, context)

    assert result["status"] == "INVALID_REQUEST"
    assert "strategy_id" in result["message"]
    assert "factors" in result["message"]


def test_run_backtest_blocks_when_factor_required_columns_are_missing(registry, lake):
    _write_multi_factor_bars(lake)
    context = ToolContext(run_id="strategy-missing-factor-columns")

    result = registry.run_tool(
        "run_backtest",
        {
            "factor_name": "pb_rank",
            "start_date": "20240101",
            "end_date": "20240215",
            "symbols": ["000001.SZ", "000002.SZ", "000003.SZ"],
            "top_n": 1,
        },
        context,
    )

    assert result["status"] == "BLOCKED"
    assert result["reason"] == "MISSING_FACTOR_INPUTS"
    assert result["factor_id"] == "pb_rank"
    assert result["missing_columns"] == ["pb"]
    assert result["coverage_status"] == "NO_DATA"
    assert result["next_repair_tool"] == "run_tushare_fetch"


def test_generate_research_report_includes_evidence_limitations_and_gaps(registry):
    context = ToolContext(run_id="research-report", experiment_id="exp_report")

    result = registry.run_tool(
        "generate_research_report",
        {
            "experiment_id": "exp_report",
            "run_ids": ["research_missing"],
            "include_sections": ["summary", "metrics", "limitations", "data_gaps"],
        },
        context,
    )

    text = Path(result["report_path"]).read_text(encoding="utf-8")
    assert "## Evidence Status" in text
    assert "## Effective Candidates / 有效候选" in text
    assert "## Failed Candidates / 失败候选" in text
    assert "## Blocked Candidates / 阻断候选" in text
    assert "## Limitations" in text
    assert "## Data Gaps / 数据缺口" in text
    assert "## Diagnostic Gaps / 诊断缺口" in text
    assert "## Next Actions / 下一步动作" in text
    assert "research_missing" in text
    assert "No run artifact found" in text


def test_agent_can_list_legacy_flat_strategy_candidates(registry, tmp_path):
    context = ToolContext(run_id="strategy-list-flat")
    generated = tmp_path / "generated" / "strategies" / "strat_legacy.py"
    generated.parent.mkdir(parents=True, exist_ok=True)
    generated.write_text("def generate_signals(data):\n    return data\n", encoding="utf-8")
    generated.with_name("test_strat_legacy.py").write_text(
        "def test_placeholder():\n    assert True\n",
        encoding="utf-8",
    )

    listed = registry.run_tool(
        "list_strategy_candidates",
        {"query": "strat_legacy"},
        context,
    )

    assert listed["status"] == "ok"
    assert listed["count"] == 1
    assert listed["strategies"][0]["strategy_id"] == "strat_legacy"
    assert listed["strategies"][0]["tests_path"].endswith("test_strat_legacy.py")


def _write_multi_factor_bars(lake: DataLake) -> None:
    rows = []
    start = date(2024, 1, 1)
    for offset in range(46):
        trade_date = f"{start + timedelta(days=offset):%Y%m%d}"
        for symbol_index, symbol in enumerate(["000001.SZ", "000002.SZ", "000003.SZ"]):
            base = 10 + symbol_index * 5
            drift = offset * (0.2 - symbol_index * 0.04)
            rows.append(
                {
                    "ts_code": symbol,
                    "trade_date": trade_date,
                    "open": base + drift,
                    "high": base + drift + 0.5,
                    "low": base + drift - 0.5,
                    "close": base + drift + (0.1 if offset % 2 == 0 else -0.1),
                    "vol": 100000 + symbol_index * 1000,
                    "amount": 1000000 + symbol_index * 10000,
                    "turnover": 0.02 + symbol_index * 0.01,
                }
            )
    lake.write_parquet(pd.DataFrame(rows), "raw", "tushare/daily")
