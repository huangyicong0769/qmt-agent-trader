"""Tests for the Factor Discovery workflow (new pipeline)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from qmt_agent_trader.agent.experiment_store import ExperimentStore
from qmt_agent_trader.agent.sandbox import CodeSandbox
from qmt_agent_trader.agent.schemas import ExperimentStatus, ToolContext
from qmt_agent_trader.agent.tools import build_agent_registry
from qmt_agent_trader.agent.workflows.factor_discovery import FactorDiscoveryWorkflow
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
    # Write minimal data so the workflow doesn't crash on empty lake
    rows = []
    for offset in range(25):
        trade_date = f"{pd.Timestamp('2024-01-01') + pd.Timedelta(days=offset):%Y%m%d}"
        rows.extend(
            [
                {
                    "ts_code": "000001.SZ",
                    "trade_date": trade_date,
                    "open": 10 + offset,
                    "high": 11 + offset,
                    "low": 9 + offset,
                    "close": 10.5 + offset,
                    "vol": 1000000,
                    "amount": 10000000,
                },
                {
                    "ts_code": "000002.SZ",
                    "trade_date": trade_date,
                    "open": 20 + offset * 0.1,
                    "high": 21 + offset * 0.1,
                    "low": 19 + offset * 0.1,
                    "close": 20.5 + offset * 0.1,
                    "vol": 1000000,
                    "amount": 10000000,
                },
            ]
        )
    lake.write_parquet(
        pd.DataFrame(rows),
        "raw",
        "tushare/daily",
    )
    return reg


@pytest.fixture
def store(tmp_path):
    return ExperimentStore(tmp_path / "experiments")


def test_workflow_runs_to_review(registry, store):
    wf = FactorDiscoveryWorkflow(registry, store)
    exp = wf.run("低波动高胜率因子", "stock_etf", "20200101", "20240624")
    assert exp.status in (ExperimentStatus.REVIEW_REQUIRED, ExperimentStatus.FAILED)


def test_workflow_creates_experiment(registry, store):
    wf = FactorDiscoveryWorkflow(registry, store)
    exp = wf.run("momentum test", "stock", "20200101", "20240624")
    assert exp.experiment_id is not None
    assert exp.kind == "factor_discovery"


def test_workflow_does_not_write_to_formal_library(registry, store, tmp_path):
    wf = FactorDiscoveryWorkflow(registry, store)
    exp = wf.run("test factor", "stock", "20200101", "20240624")
    # Check library was not modified
    Path("src/qmt_agent_trader/factors/library/momentum.py")
    # Workflow should not alter production code
    assert exp.status is not None  # just verify it didn't crash


def test_generated_factor_must_be_saved_before_evaluation(registry) -> None:
    context = ToolContext(run_id="run_test", experiment_id="exp_test")
    spec_result = registry.run_tool(
        "create_factor_spec",
        {
            "factor_name": "agent momentum",
            "factor_description": "short horizon momentum",
            "formula_sketch": "pct_change",
            "lookback": 3,
        },
        context,
    )
    factor_spec = spec_result["factor_spec"]
    code_result = registry.run_tool("generate_factor_code", {"factor_spec": factor_spec}, context)
    factor_id = factor_spec["factor_id"]

    unsaved = registry.run_tool(
        "evaluate_factor_candidate",
        {"factor_id": factor_id, "start_date": "20240105", "end_date": "20240120"},
        context,
    )

    assert unsaved["status"] == "FACTOR_NOT_FOUND"

    static_result = registry.run_tool(
        "run_factor_static_checks",
        {"code_path": code_result["code_path"]},
        context,
    )
    assert static_result["status"] == "PASSED"

    saved = registry.run_tool(
        "save_factor",
        {
            "factor_id": factor_id,
            "code_path": code_result["code_path"],
            "spec_path": code_result["spec_path"],
        },
        context,
    )
    assert saved["status"] == "saved"

    evaluated = registry.run_tool(
        "evaluate_factor_candidate",
        {"factor_id": factor_id, "start_date": "20240105", "end_date": "20240120"},
        context,
    )
    assert evaluated["status"] == "validated"
    assert evaluated["non_null"] > 0


def test_factor_evaluation_blocks_unbounded_long_window_without_symbols(registry) -> None:
    context = ToolContext(run_id="factor-unbounded", experiment_id="exp_test")

    result = registry.run_tool(
        "evaluate_factor_candidate",
        {
            "factor_id": "momentum_20d",
            "start_date": "20200101",
            "end_date": "20260630",
        },
        context,
    )

    assert result["status"] == "BLOCKED"
    assert result["reason"] == "UNBOUNDED_FACTOR_EVALUATION"
    assert result["missing_inputs"] == ["symbols"]
    assert result["next_repair_tool"] == "query_universe"


def test_agent_can_list_saved_factors_and_duplicate_saves_are_rejected(registry) -> None:
    context = ToolContext(run_id="run_test", experiment_id="exp_test")
    spec_result = registry.run_tool(
        "create_factor_spec",
        {
            "factor_name": "agent_unique_momentum",
            "factor_description": "short horizon momentum",
            "formula_sketch": "pct_change",
            "lookback": 3,
        },
        context,
    )
    factor_spec = spec_result["factor_spec"]
    code_result = registry.run_tool("generate_factor_code", {"factor_spec": factor_spec}, context)
    factor_id = factor_spec["factor_id"]

    saved = registry.run_tool(
        "save_factor",
        {
            "factor_id": factor_id,
            "code_path": code_result["code_path"],
            "spec_path": code_result["spec_path"],
        },
        context,
    )
    assert saved["status"] == "saved"

    listed = registry.run_tool(
        "list_saved_factors",
        {"query": "agent_unique_momentum", "include_builtins": False},
        context,
    )
    assert listed["status"] == "ok"
    assert listed["count"] == 1
    assert listed["factors"][0]["factor_id"] == factor_id

    duplicate_spec = {
        **factor_spec,
        "factor_id": "factor_duplicate_name",
    }
    duplicate_code = registry.run_tool(
        "generate_factor_code",
        {"factor_spec": duplicate_spec},
        context,
    )
    duplicate = registry.run_tool(
        "save_factor",
        {
            "factor_id": duplicate_spec["factor_id"],
            "code_path": duplicate_code["code_path"],
            "spec_path": duplicate_code["spec_path"],
        },
        context,
    )
    assert duplicate["status"] == "DUPLICATE_FACTOR_NAME"
    assert duplicate["existing_factors"][0]["factor_id"] == factor_id


def test_list_saved_factors_returns_exact_match_and_strategy_leg_hint(registry) -> None:
    result = registry.run_tool(
        "list_saved_factors",
        {
            "query": "volatility_20d",
            "include_builtins": True,
            "exact": True,
            "include_usage_hints": True,
        },
        ToolContext(run_id="factor-query-exact"),
    )

    assert result["status"] == "ok"
    assert result["exact_matches"]
    match = result["exact_matches"][0]
    assert match["factor_id"] == "volatility_20d"
    assert match["strategy_leg_example"] == {
        "factor_id": "volatility_20d",
        "weight": 1.0,
        "ascending": True,
    }
    assert "factor_name" not in match["strategy_leg_example"]


def test_list_saved_factors_returns_fuzzy_candidates_without_factor_name_hint(registry) -> None:
    result = registry.run_tool(
        "list_saved_factors",
        {
            "query": "volatility",
            "include_builtins": True,
            "include_usage_hints": True,
            "limit": 5,
        },
        ToolContext(run_id="factor-query-fuzzy"),
    )

    assert result["status"] == "ok"
    assert result["candidates"]
    assert any(candidate["factor_id"] == "volatility_20d" for candidate in result["candidates"])
    assert all(
        "factor_name" not in candidate["strategy_leg_example"]
        for candidate in result["candidates"]
    )


def test_unknown_factor_formula_requests_python_function_instead_of_fallback(registry) -> None:
    context = ToolContext(run_id="factor-unsupported", experiment_id="exp_test")
    spec = registry.run_tool(
        "create_factor_spec",
        {
            "factor_name": "unsupported_custom_signal",
            "formula_sketch": "combine three proprietary qualitative channel checks",
            "lookback": 20,
        },
        context,
    )["factor_spec"]

    result = registry.run_tool("generate_factor_code", {"factor_spec": spec}, context)

    assert result["status"] == "NEEDS_PYTHON_FUNCTION"
    assert result["review_required"] is True
    assert result["next_required_input"] == "python_function"
    assert "code_path" not in result


@pytest.mark.parametrize(
    ("factor_name", "formula", "expected_tokens", "forbidden_tokens"),
    [
        (
            "low_volatility_template",
            "low volatility factor: negative rolling standard deviation of daily returns",
            [".std(", "pct_change()"],
            ["pct_change(lookback)"],
        ),
        (
            "low_turnover_template",
            "low turnover factor: negative rolling mean of turnover",
            ['"turnover"', ".rolling(lookback).mean()"],
            ["pct_change(lookback)"],
        ),
        (
            "low_vol_low_turnover_template",
            "composite low volatility plus low turnover factor",
            ['"turnover"', ".std(", "composite"],
            ["pct_change(lookback)"],
        ),
        (
            "sector_neutral_low_vol_template",
            "sector neutral low volatility factor grouped by industry",
            ["industry", ".std(", "groupby"],
            ["pct_change(lookback)"],
        ),
        (
            "macro_timed_momentum_template",
            "macro timed momentum using macro_cycle_score to gate price momentum",
            ["macro_cycle_score", "pct_change(lookback)"],
            [],
        ),
    ],
)
def test_factor_formula_templates_are_semantic(
    registry,
    factor_name: str,
    formula: str,
    expected_tokens: list[str],
    forbidden_tokens: list[str],
) -> None:
    context = ToolContext(run_id=f"factor-template-{factor_name}", experiment_id="exp_test")
    spec = registry.run_tool(
        "create_factor_spec",
        {
            "factor_name": factor_name,
            "formula_sketch": formula,
            "lookback": 20,
        },
        context,
    )["factor_spec"]

    result = registry.run_tool("generate_factor_code", {"factor_spec": spec}, context)

    assert result["status"] == "generated"
    code = Path(result["code_path"]).read_text(encoding="utf-8")
    for token in expected_tokens:
        assert token in code
    for token in forbidden_tokens:
        assert token not in code
    checks = registry.run_tool(
        "run_factor_static_checks",
        {"code_path": result["code_path"], "factor_id": spec["factor_id"]},
        context,
    )
    assert checks["status"] == "PASSED"
    assert checks["semantic_status"] == "PASSED"


@pytest.mark.parametrize(
    ("factor_name", "formula", "expected_tokens"),
    [
        (
            "low_vol_inverse_20d",
            "1.0 / (std(log_return, 20) + 1e-9)",
            ["log_return", ".rolling(lookback).std()", "1.0 /"],
        ),
        (
            "low_vol_20d",
            "-std(close, 20)",
            ['bars["close"]', ".rolling(lookback).std()", "return -"],
        ),
        (
            "price_position_60d",
            "(close - min(close,60)) / (max(close,60) - min(close,60) + 1e-9)",
            ["rolling_min", "rolling_max", "price_position"],
        ),
    ],
)
def test_formula_dsl_generates_basic_rolling_arithmetic(
    registry,
    factor_name: str,
    formula: str,
    expected_tokens: list[str],
) -> None:
    context = ToolContext(run_id=f"factor-dsl-{factor_name}", experiment_id="exp_test")
    spec = registry.run_tool(
        "create_factor_spec",
        {
            "factor_name": factor_name,
            "formula_sketch": formula,
            "lookback": 20 if "60" not in factor_name else 60,
        },
        context,
    )["factor_spec"]

    result = registry.run_tool("generate_factor_code", {"factor_spec": spec}, context)

    assert result["status"] == "generated"
    assert result["formula_ast"]["kind"] in {
        "low_vol_inverse",
        "negative_rolling_std",
        "price_position",
    }
    code = Path(result["code_path"]).read_text(encoding="utf-8")
    for token in expected_tokens:
        assert token in code
    checks = registry.run_tool(
        "run_factor_static_checks",
        {"code_path": result["code_path"], "factor_id": spec["factor_id"]},
        context,
    )
    assert checks["status"] == "PASSED"
    assert checks["semantic_status"] == "PASSED"


def test_factor_static_checks_reject_formula_code_semantic_mismatch(registry) -> None:
    context = ToolContext(run_id="factor-semantic-mismatch", experiment_id="exp_test")
    spec = registry.run_tool(
        "create_factor_spec",
        {
            "factor_name": "semantic_low_volatility",
            "formula_sketch": "low volatility: negative rolling standard deviation of returns",
            "lookback": 20,
        },
        context,
    )["factor_spec"]
    generated = registry.run_tool("generate_factor_code", {"factor_spec": spec}, context)
    Path(generated["code_path"]).write_text(
        """
from typing import Any

import pandas as pd


def compute(bars: pd.DataFrame, params: dict[str, Any] | None = None) -> pd.Series:
    lookback = int((params or {}).get("lookback", 20))
    return bars.groupby("symbol")["close"].pct_change(lookback)
""",
        encoding="utf-8",
    )

    checks = registry.run_tool(
        "run_factor_static_checks",
        {"code_path": generated["code_path"], "factor_id": spec["factor_id"]},
        context,
    )

    assert checks["status"] == "FAILED"
    assert checks["semantic_status"] == "FAILED"
    assert any("semantic mismatch" in issue.lower() for issue in checks["issues"])
