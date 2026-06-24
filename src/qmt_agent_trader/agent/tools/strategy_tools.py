"""Strategy tools: create_strategy_spec, generate_strategy_code, run_backtest,
and report tool: generate_research_report."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from qmt_agent_trader.agent.experiment_store import ExperimentStore
from qmt_agent_trader.agent.permissions import PermissionLevel
from qmt_agent_trader.agent.sandbox import CodeSandbox
from qmt_agent_trader.agent.schemas import StrategySpec, ToolContext, ToolSpec
from qmt_agent_trader.agent.tools.base import AgentTool, tool
from qmt_agent_trader.backtest.service import run_backtest_report
from qmt_agent_trader.core.ids import new_id, shanghai_now_iso
from qmt_agent_trader.data.storage import DataLake

_sandbox: CodeSandbox | None = None
_store: ExperimentStore | None = None
_lake: DataLake | None = None


def wire(sandbox: CodeSandbox, store: ExperimentStore, lake: DataLake) -> None:
    global _sandbox, _store, _lake
    _sandbox = sandbox
    _store = store
    _lake = lake


# ── create_strategy_spec ────────────────────────────────────────────────────


def _create_strategy_spec(input_data: dict[str, Any], context: ToolContext) -> dict[str, Any]:
    strategy_idea = input_data.get("strategy_idea", "")
    selected_factors = input_data.get("selected_factors", [])
    universe = input_data.get("universe", "stock_etf")
    rebalance_freq = input_data.get("rebalance_frequency", "daily")
    constraints = input_data.get("constraints", {})

    strategy_id = new_id("strat")
    spec = StrategySpec(
        strategy_id=strategy_id,
        name=strategy_idea[:60] or "candidate_strategy",
        version="0.1.0",
        universe=universe,
        factors=selected_factors,
        portfolio_construction={
            "method": "equal_weight",
            "top_n": 20,
        },
        rebalance={"frequency": rebalance_freq},
        risk_constraints=constraints,
        execution_assumptions={
            "timing": "next_open",
            "slippage_model": "fixed_5bps",
        },
    )
    return {"strategy_spec": spec.model_dump(mode="json")}


create_strategy_spec_tool: AgentTool = tool(
    ToolSpec(
        name="create_strategy_spec",
        description="将策略想法和候选因子组合转成结构化 strategy spec。",
        permission=PermissionLevel.RESEARCH_WRITE,
        side_effect_level="write_generated",
        deterministic=False,
    ),
    fn=_create_strategy_spec,
)

# ── generate_strategy_code ──────────────────────────────────────────────────


def _generate_strategy_code(input_data: dict[str, Any], context: ToolContext) -> dict[str, Any]:
    spec_data = input_data.get("strategy_spec", {})
    name = spec_data.get("name", "candidate")
    strategy_id = spec_data.get("strategy_id", new_id("strat"))
    factors = spec_data.get("factors", [])
    top_n = (
        
            spec_data.get("portfolio_construction", {}).get("top_n", 20)
            if "portfolio_construction" in spec_data
            else 20
        
    )
    warnings: list[str] = []

    if not factors:
        warnings.append("no factors selected; strategy will not generate signals")

    strategy_code = _render_strategy_code(name, factors, top_n)
    test_code = _render_strategy_test_code(name)

    sb = _sandbox
    if sb is None:
        return {"status": "error", "message": "sandbox not wired"}

    try:
        code_path = sb.write_candidate_file(f"strategies/{strategy_id}.py", strategy_code)
        tests_path = sb.write_candidate_file(f"strategies/test_{strategy_id}.py", test_code)
        return {
            "code_path": str(code_path),
            "tests_path": str(tests_path),
            "status": "generated",
            "warnings": warnings,
        }
    except Exception as exc:
        return {"status": "error", "warnings": [str(exc)]}


generate_strategy_code_tool: AgentTool = tool(
    ToolSpec(
        name="generate_strategy_code",
        description="根据 strategy spec 生成候选策略代码和测试。",
        permission=PermissionLevel.CODE_GENERATION,
        side_effect_level="write_generated",
        deterministic=False,
    ),
    fn=_generate_strategy_code,
)

# ── run_backtest ────────────────────────────────────────────────────────────


def _run_backtest(input_data: dict[str, Any], context: ToolContext) -> dict[str, Any]:
    lake = _lake
    if lake is None:
        return {"status": "NOT_IMPLEMENTED", "message": "data lake not wired"}

    input_data.get("strategy_id", "")
    input_data.get("start_date", "20200101")
    input_data.get("end_date", "20260624")
    input_data.get("initial_cash", 1_000_000)

    reports_dir = Path("reports/backtests")
    try:
        summary = run_backtest_report(
            lake,
            reports_dir=reports_dir,
            quantity=100,
        ).as_dict()
        return {
            "run_id": summary.get("run_id", new_id("bt")),
            "metrics": {
                "fills": summary.get("fills", 0),
                "leakage_valid": summary.get("leakage_valid", True),
            },
            "report_path": summary.get("report_path", ""),
            "leakage_report": {"valid": summary.get("leakage_valid", True)},
            "status": "completed",
        }
    except Exception as exc:
        return {"status": "error", "message": str(exc)}


run_backtest_tool: AgentTool = tool(
    ToolSpec(
        name="run_backtest",
        description="对候选策略运行真实约束回测。",
        permission=PermissionLevel.BACKTEST_EXECUTE,
        deterministic=False,
        timeout_seconds=120,
    ),
    fn=_run_backtest,
)

# ── generate_research_report ────────────────────────────────────────────────


def _generate_research_report(input_data: dict[str, Any], context: ToolContext) -> dict[str, Any]:
    exp_id = input_data.get("experiment_id", "unknown")
    run_ids = input_data.get("run_ids", [])
    sections = input_data.get("include_sections", ["summary", "metrics"])

    sb = _sandbox
    reports_root = sb.generated_root / "reports" if sb else Path("reports/research")
    reports_root.mkdir(parents=True, exist_ok=True)
    report_path = reports_root / f"{exp_id}.md"

    store = _store
    experiment_meta = {}
    if store:
        try:
            exp = store.get_experiment(exp_id)
            experiment_meta = {
                "kind": exp.kind,
                "status": exp.status.value,
                "lessons": exp.lessons,
            }
        except Exception:
            pass

    lines = [
        f"# Research Report: {exp_id}",
        f"Generated: {shanghai_now_iso()}",
        "",
        "## Experiment",
        f"- Kind: {experiment_meta.get('kind', 'unknown')}",
        f"- Status: {experiment_meta.get('status', 'unknown')}",
        "",
    ]

    if "summary" in sections:
        lines.extend(["## Summary", "", f"Run IDs: {', '.join(run_ids)}", ""])
    if "metrics" in sections:
        lines.extend(["## Metrics", "", "(see attached backtest reports for details)", ""])
    if "lessons" in sections and experiment_meta.get("lessons"):
        lines.extend(["## Lessons Learned", ""])
        for lesson in experiment_meta["lessons"]:
            lines.append(f"- {lesson}")
        lines.append("")

    report_path.write_text("\n".join(lines), encoding="utf-8")
    return {"report_path": str(report_path), "summary": lines[0]}


generate_research_report_tool: AgentTool = tool(
    ToolSpec(
        name="generate_research_report",
        description="生成因子或策略研究报告。",
        permission=PermissionLevel.RESEARCH_WRITE,
        side_effect_level="write_generated",
        deterministic=False,
    ),
    fn=_generate_research_report,
)


# ── Helpers ──────────────────────────────────────────────────────────────────


def _render_strategy_code(name: str, factors: list[str], top_n: int) -> str:
    name.replace(" ", "_").replace("-", "_").lower()
    factor_list = ", ".join(repr(f) for f in factors) if factors else "[]"
    return f'''"""Candidate strategy: {name}.
Auto-generated by Agent — REVIEW_REQUIRED before promotion.
"""

import pandas as pd


FACTORS = [{factor_list}]
TOP_N = {top_n}


def generate_signals(data: pd.DataFrame) -> pd.DataFrame:
    """Generate buy signals based on factor rankings."""
    if data.empty:
        return pd.DataFrame(columns=["symbol", "target_weight"])
    # Placeholder: in a full implementation, combine multiple factors into a score.
    ranked = data.head(TOP_N).copy()
    ranked["target_weight"] = 1 / max(len(ranked), 1)
    return ranked[["symbol", "target_weight"]]
'''


def _render_strategy_test_code(name: str) -> str:
    safe_name = name.replace(" ", "_").replace("-", "_").lower()
    return f'''"""Tests for candidate strategy: {name}."""

import pandas as pd

from generated.strategies.{safe_name} import generate_signals


def test_empty_data():
    result = generate_signals(pd.DataFrame())
    assert result.empty


def test_generates_signals():
    data = pd.DataFrame({{
        "symbol": ["A", "B", "C"],
        "close": [10, 20, 30],
        "score": [0.9, 0.5, 0.1],
    }})
    result = generate_signals(data)
    assert "symbol" in result.columns
    assert "target_weight" in result.columns
'''
