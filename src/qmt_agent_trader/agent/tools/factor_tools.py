"""Factor tools: create_factor_spec, generate_factor_code, run_factor_static_checks,
evaluate_factor_candidate."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from qmt_agent_trader.agent.experiment_store import ExperimentStore
from qmt_agent_trader.agent.permissions import PermissionLevel
from qmt_agent_trader.agent.sandbox import CodeSandbox
from qmt_agent_trader.agent.schemas import FactorSpec, ToolContext, ToolSpec
from qmt_agent_trader.agent.tools.base import AgentTool, tool
from qmt_agent_trader.core.ids import new_id
from qmt_agent_trader.data.storage import DataLake
from qmt_agent_trader.factors.service import (
    validate_factor,
)

_sandbox: CodeSandbox | None = None
_store: ExperimentStore | None = None
_lake: DataLake | None = None


def wire(sandbox: CodeSandbox, store: ExperimentStore, lake: DataLake) -> None:
    global _sandbox, _store, _lake
    _sandbox = sandbox
    _store = store
    _lake = lake


# ── create_factor_spec ──────────────────────────────────────────────────────


def _create_factor_spec(input_data: dict[str, Any], context: ToolContext) -> dict[str, Any]:
    hypothesis = input_data.get("hypothesis", {})
    name = hypothesis.get("name", "unnamed")
    hypothesis.get("intuition", "")
    required_data = hypothesis.get("required_data", ["daily_bars"])
    formula_sketch = hypothesis.get("formula_sketch", "")
    lookback = hypothesis.get("expected_holding_period", "20")
    try:
        lookback = int(str(lookback).replace("d", "").strip())
    except ValueError:
        lookback = 20

    factor_id = new_id("factor")
    spec = FactorSpec(
        factor_id=factor_id,
        name=name,
        version="0.1.0",
        inputs=required_data,
        lookback=lookback,
        formula=formula_sketch,
    )
    return {"factor_spec": spec.model_dump(mode="json")}


create_factor_spec_tool: AgentTool = tool(
    ToolSpec(
        name="create_factor_spec",
        description="将自然语言因子假设转成结构化 factor spec。",
        permission=PermissionLevel.RESEARCH_WRITE,
        side_effect_level="write_generated",
        deterministic=False,
    ),
    fn=_create_factor_spec,
)

# ── generate_factor_code ────────────────────────────────────────────────────


def _generate_factor_code(input_data: dict[str, Any], context: ToolContext) -> dict[str, Any]:
    spec_data = input_data.get("factor_spec", {})
    name = spec_data.get("name", "candidate")
    factor_id = spec_data.get("factor_id", new_id("factor"))
    lookback = spec_data.get("lookback", 20)
    formula = spec_data.get("formula", "return")
    warnings: list[str] = []

    factor_code = _render_factor_code(name, lookback, formula)
    test_code = _render_factor_test_code(name)

    sb = _sandbox
    if sb is None:
        return {"status": "error", "message": "sandbox not wired"}

    try:
        code_path = sb.write_candidate_file(f"factors/{factor_id}.py", factor_code)
        tests_path = sb.write_candidate_file(f"factors/test_{factor_id}.py", test_code)
        return {
            "code_path": str(code_path),
            "tests_path": str(tests_path),
            "status": "generated",
            "warnings": warnings,
        }
    except Exception as exc:
        return {"status": "error", "warnings": [str(exc)]}


generate_factor_code_tool: AgentTool = tool(
    ToolSpec(
        name="generate_factor_code",
        description="根据 factor spec 生成候选因子代码和测试。",
        permission=PermissionLevel.CODE_GENERATION,
        side_effect_level="write_generated",
        deterministic=False,
    ),
    fn=_generate_factor_code,
)

# ── run_factor_static_checks ────────────────────────────────────────────────


def _run_factor_static_checks(input_data: dict[str, Any], context: ToolContext) -> dict[str, Any]:
    code_path_str = input_data.get("code_path", "")
    code_path = Path(code_path_str)
    if not code_path.exists():
        return {"status": "FAILED", "issues": [f"file not found: {code_path_str}"]}
    sb = _sandbox
    if sb is None:
        return {"status": "FAILED", "issues": ["sandbox not wired"]}
    issues = sb.static_scan_code(code_path.read_text(encoding="utf-8"))
    return {"status": "PASSED" if not issues else "FAILED", "issues": issues}


run_factor_static_checks_tool: AgentTool = tool(
    ToolSpec(
        name="run_factor_static_checks",
        description="检查候选因子是否存在未来函数或危险行为。",
        permission=PermissionLevel.BACKTEST_EXECUTE,
        deterministic=True,
    ),
    fn=_run_factor_static_checks,
)

# ── evaluate_factor_candidate ────────────────────────────────────────────────


def _evaluate_factor_candidate(input_data: dict[str, Any], context: ToolContext) -> dict[str, Any]:
    factor_id = input_data.get("factor_id", "")
    start = input_data.get("start_date", "20200101")
    end = input_data.get("end_date", "20260624")

    lake = _lake
    if lake is None:
        return {"status": "NOT_IMPLEMENTED", "message": "data lake not wired"}

    # Map factor_id to built-in name or stub
    factor_name = _map_factor_name(factor_id)
    if factor_name is None:
        return {
            "status": "NOT_IMPLEMENTED",
            "message": f"factor '{factor_id}' not in built-in set; use generate_factor_code first",
        }

    try:
        result = validate_factor(lake, name=factor_name, start=start, end=end).as_dict()
        # Compute quantile returns via walk-forward
        from qmt_agent_trader.factors.service import walk_forward_factor_validation

        wf = walk_forward_factor_validation(
            lake, name=factor_name, start=start, end=end
        ).as_dict()
        slices_raw = wf.get("walk_forward", [])
        if not isinstance(slices_raw, list):
            slices_raw = []
        spreads: list[float] = []
        for s in slices_raw:
            if isinstance(s, dict):
                val = s.get("long_short_spread", 0)
                if isinstance(val, (int, float)):
                    spreads.append(float(val))
        result["quantile_returns"] = {
            "long_short_spread_mean": sum(spreads) / len(spreads) if spreads else 0,
            "walk_forward_slices": len(slices_raw),
        }
        result["report_path"] = str(
            _sandbox.generated_root / "reports" / f"factor_{factor_id}.json"
            if _sandbox
            else ""
        )
        return result
    except Exception as exc:
        return {"status": "error", "message": str(exc)}


evaluate_factor_candidate_tool: AgentTool = tool(
    ToolSpec(
        name="evaluate_factor_candidate",
        description="计算并评估候选因子。",
        permission=PermissionLevel.BACKTEST_EXECUTE,
        deterministic=False,
        timeout_seconds=120,
    ),
    fn=_evaluate_factor_candidate,
)

# ── Helpers ──────────────────────────────────────────────────────────────────

_BUILTIN_FACTORS = {
    "momentum_20d": "momentum_20d",
    "momentum_60d": "momentum_60d",
    "reversal_5d": "reversal_5d",
    "volatility_20d": "volatility_20d",
    "turnover_20d": "turnover_20d",
    "amount_zscore_20d": "amount_zscore_20d",
}


def _map_factor_name(factor_id: str) -> str | None:
    if factor_id in _BUILTIN_FACTORS:
        return _BUILTIN_FACTORS[factor_id]
    for key, value in _BUILTIN_FACTORS.items():
        if key in factor_id or factor_id in key:
            return value
    return None


def _render_factor_code(name: str, lookback: int, formula: str) -> str:
    name.replace(" ", "_").replace("-", "_").lower()
    return f'''"""Candidate factor: {name}.
Auto-generated by Agent — REVIEW_REQUIRED before promotion.
"""

import pandas as pd


def compute(bars: pd.DataFrame) -> pd.Series:
    """{formula}

    Lookback: {lookback} days.
    """
    return bars.groupby("symbol")["close"].pct_change({lookback})
'''


def _render_factor_test_code(name: str) -> str:
    safe_name = name.replace(" ", "_").replace("-", "_").lower()
    return f'''"""Tests for candidate factor: {name}."""

import pandas as pd

from generated.factors.{safe_name} import compute


def test_compute_returns_series():
    bars = pd.DataFrame({{
        "symbol": ["A", "A", "A", "B", "B", "B"],
        "trade_date": pd.date_range("2024-01-01", periods=6),
        "close": [10, 11, 12, 20, 21, 22],
    }})
    result = compute(bars)
    assert isinstance(result, pd.Series)
    assert len(result) == len(bars)
    assert result.isna().sum() > 0  # lookback creates NaN


def test_empty_input():
    result = compute(pd.DataFrame(columns=["symbol", "trade_date", "close"]))
    assert len(result) == 0
'''
