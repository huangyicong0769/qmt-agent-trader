"""Factor tools for factor specs, drafts, saved registry entries, and evaluation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from qmt_agent_trader.agent.experiment_store import ExperimentStore
from qmt_agent_trader.agent.permissions import PermissionLevel
from qmt_agent_trader.agent.sandbox import CodeSandbox
from qmt_agent_trader.agent.schemas import FactorSpec, ToolContext, ToolSpec
from qmt_agent_trader.agent.tools.base import AgentTool, tool
from qmt_agent_trader.core.ids import new_id
from qmt_agent_trader.data.storage import DataLake
from qmt_agent_trader.factors.registry import FactorRegistry
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
    hypothesis = input_data.get("hypothesis")
    if isinstance(hypothesis, dict):
        hypothesis = {**hypothesis, **{k: v for k, v in input_data.items() if k != "hypothesis"}}
    else:
        hypothesis = input_data
    name = hypothesis.get("name", "unnamed")
    name = hypothesis.get("factor_name", name)
    required_data = hypothesis.get("required_data") or hypothesis.get("data_sources") or [
        "daily_bars"
    ]
    formula_sketch = hypothesis.get("formula_sketch") or hypothesis.get("factor_description", "")
    lookback = hypothesis.get("lookback", hypothesis.get("expected_holding_period", "20"))
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
        input_schema={
            "type": "object",
            "properties": {
                "hypothesis": {"type": "object"},
                "factor_name": {"type": "string"},
                "factor_description": {"type": "string"},
                "formula_sketch": {"type": "string"},
                "lookback": {"type": "integer"},
                "data_sources": {"type": "array", "items": {"type": "string"}},
                "universe": {"type": "string"},
            },
        },
        permission=PermissionLevel.RESEARCH_WRITE,
        side_effect_level="write_generated",
        deterministic=False,
    ),
    fn=_create_factor_spec,
)

# ── generate_factor_code ────────────────────────────────────────────────────


def _generate_factor_code(input_data: dict[str, Any], context: ToolContext) -> dict[str, Any]:
    spec_data = input_data.get("factor_spec", {})
    if not isinstance(spec_data, dict) or not spec_data.get("factor_id"):
        return {
            "status": "INVALID_REQUEST",
            "message": "factor_spec with factor_id is required",
        }
    name = spec_data.get("name", "candidate")
    factor_id = spec_data.get("factor_id", new_id("factor"))
    lookback = spec_data.get("lookback", 20)
    formula = spec_data.get("formula", "return")
    warnings: list[str] = []

    factor_code = _render_factor_code(name, lookback, formula)
    test_code = _render_factor_test_code(name)
    spec_code = json.dumps(spec_data, ensure_ascii=False, indent=2, default=str)

    sb = _sandbox
    if sb is None:
        return {"status": "error", "message": "sandbox not wired"}

    try:
        code_path = sb.write_candidate_file(f"factors/drafts/{factor_id}/factor.py", factor_code)
        tests_path = sb.write_candidate_file(
            f"factors/drafts/{factor_id}/test_factor.py",
            test_code,
        )
        spec_path = sb.write_candidate_file(
            f"factors/drafts/{factor_id}/factor_spec.json",
            spec_code,
        )
        return {
            "factor_id": factor_id,
            "code_path": str(code_path),
            "tests_path": str(tests_path),
            "spec_path": str(spec_path),
            "status": "generated",
            "warnings": warnings,
        }
    except Exception as exc:
        return {"status": "error", "warnings": [str(exc)]}


generate_factor_code_tool: AgentTool = tool(
    ToolSpec(
        name="generate_factor_code",
        description="根据 factor spec 生成候选因子代码和测试。",
        input_schema={
            "type": "object",
            "properties": {"factor_spec": {"type": "object"}},
            "required": ["factor_spec"],
        },
        permission=PermissionLevel.CODE_GENERATION,
        side_effect_level="write_generated",
        deterministic=False,
    ),
    fn=_generate_factor_code,
)

# ── run_factor_static_checks ────────────────────────────────────────────────


def _run_factor_static_checks(input_data: dict[str, Any], context: ToolContext) -> dict[str, Any]:
    code_path_str = input_data.get("code_path", "")
    if not code_path_str:
        return {"status": "INVALID_REQUEST", "issues": ["code_path is required"]}
    code_path = Path(code_path_str)
    if not code_path.exists():
        return {"status": "FAILED", "issues": [f"file not found: {code_path_str}"]}
    if not code_path.is_file():
        return {"status": "INVALID_REQUEST", "issues": [f"not a file: {code_path_str}"]}
    sb = _sandbox
    if sb is None:
        return {"status": "FAILED", "issues": ["sandbox not wired"]}
    issues = sb.static_scan_code(code_path.read_text(encoding="utf-8"))
    return {"status": "PASSED" if not issues else "FAILED", "issues": issues}


run_factor_static_checks_tool: AgentTool = tool(
    ToolSpec(
        name="run_factor_static_checks",
        description="检查候选因子是否存在未来函数或危险行为。",
        input_schema={
            "type": "object",
            "properties": {"code_path": {"type": "string"}},
            "required": ["code_path"],
        },
        permission=PermissionLevel.BACKTEST_EXECUTE,
        deterministic=True,
    ),
    fn=_run_factor_static_checks,
)

# ── save_factor ──────────────────────────────────────────────────────────────


def _save_factor(input_data: dict[str, Any], context: ToolContext) -> dict[str, Any]:
    sb = _sandbox
    lake = _lake
    if sb is None:
        return {"status": "error", "message": "sandbox not wired"}
    if lake is None:
        return {"status": "error", "message": "data lake not wired"}

    factor_id = str(input_data.get("factor_id") or "").strip()
    code_path_raw = str(input_data.get("code_path") or "").strip()
    spec_path_raw = str(input_data.get("spec_path") or "").strip()
    if not factor_id:
        return {"status": "INVALID_REQUEST", "message": "factor_id is required"}
    if not code_path_raw:
        return {"status": "INVALID_REQUEST", "message": "code_path is required"}

    try:
        code_path = sb.validate_path(code_path_raw)
    except Exception as exc:
        return {"status": "INVALID_REQUEST", "message": str(exc)}
    if not code_path.exists():
        return {"status": "INVALID_REQUEST", "message": f"file not found: {code_path}"}

    issues = sb.static_scan_code(code_path.read_text(encoding="utf-8"))
    if issues:
        return {"status": "FAILED", "issues": issues}

    spec_data = _load_factor_spec(spec_path_raw, sb) if spec_path_raw else {}
    if spec_data.get("factor_id") and spec_data["factor_id"] != factor_id:
        return {
            "status": "INVALID_REQUEST",
            "message": "factor_id does not match factor_spec",
        }
    registry = FactorRegistry(_factor_registry_root(lake))
    saved = registry.save_factor(
        factor_id=factor_id,
        name=str(spec_data.get("name") or factor_id),
        version=str(spec_data.get("version") or "0.1.0"),
        implementation_ref=f"file:{code_path}",
        required_columns=_required_columns_for_spec(spec_data),
        lookback=int(spec_data.get("lookback") or 20),
        params={"lookback": int(spec_data.get("lookback") or 20)},
        created_by=str(input_data.get("created_by") or "agent"),
    )
    return {
        "status": saved.status,
        "factor_id": saved.factor_id,
        "registry_path": str(registry.registry_path),
        "implementation_ref": saved.implementation_ref,
    }


save_factor_tool: AgentTool = tool(
    ToolSpec(
        name="save_factor",
        description=(
            "将通过静态检查的候选因子保存到统一 Factor Registry。保存后才可评估或回测。"
        ),
        input_schema={
            "type": "object",
            "properties": {
                "factor_id": {"type": "string"},
                "code_path": {"type": "string"},
                "spec_path": {"type": "string"},
                "created_by": {"type": "string"},
            },
            "required": ["factor_id", "code_path"],
        },
        permission=PermissionLevel.RESEARCH_WRITE,
        side_effect_level="write_generated",
        deterministic=False,
    ),
    fn=_save_factor,
)

# ── evaluate_factor_candidate ────────────────────────────────────────────────


def _evaluate_factor_candidate(input_data: dict[str, Any], context: ToolContext) -> dict[str, Any]:
    factor_id = input_data.get("factor_id", "")
    if not str(factor_id).strip():
        return {"status": "INVALID_REQUEST", "message": "factor_id is required"}
    start = input_data.get("start_date", "20200101")
    end = input_data.get("end_date", "20260624")
    symbols = _requested_symbols(input_data)

    lake = _lake
    if lake is None:
        return {"status": "NOT_IMPLEMENTED", "message": "data lake not wired"}

    factor_name = str(factor_id).strip()
    registry_root = _factor_registry_root(lake)
    registry = FactorRegistry(registry_root)
    if registry.get_factor(factor_name) is None:
        return {
            "status": "FACTOR_NOT_SAVED",
            "message": f"factor '{factor_name}' is a draft or unknown; save_factor first",
        }

    # ── Dedup: check cache first ──
    from qmt_agent_trader.agent.tools.cache import (
        get_cached_validation,
        put_cached_validation,
    )
    cache_factor_name = factor_name if not symbols else f"{factor_name}|{','.join(symbols)}"
    cached = get_cached_validation(cache_factor_name, start, end)
    if cached is not None:
        cached["cache_hit"] = True
        return cached

    try:
        result = validate_factor(
            lake,
            name=factor_name,
            start=start,
            end=end,
            registry_root=str(registry_root),
            symbols=symbols or None,
        ).as_dict()
        # Compute quantile returns via walk-forward
        from qmt_agent_trader.factors.service import walk_forward_factor_validation

        wf = walk_forward_factor_validation(
            lake,
            name=factor_name,
            start=start,
            end=end,
            registry_root=str(registry_root),
            symbols=symbols or None,
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
        result["cache_hit"] = False
        put_cached_validation(cache_factor_name, start, end, result)
        return result
    except Exception as exc:
        return {"status": "error", "message": str(exc)}


evaluate_factor_candidate_tool: AgentTool = tool(
    ToolSpec(
        name="evaluate_factor_candidate",
        description="计算并评估候选因子。",
        input_schema={
            "type": "object",
            "properties": {
                "factor_id": {"type": "string"},
                "start_date": {"type": "string"},
                "end_date": {"type": "string"},
                "symbol": {"type": "string"},
                "code": {"type": "string"},
                "symbols": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["factor_id"],
        },
        permission=PermissionLevel.BACKTEST_EXECUTE,
        deterministic=False,
        timeout_seconds=120,
    ),
    fn=_evaluate_factor_candidate,
)

# ── Helpers ──────────────────────────────────────────────────────────────────


def _factor_registry_root(lake: DataLake) -> Path:
    return lake.root.parent / "factors"


def _load_factor_spec(spec_path_raw: str, sb: CodeSandbox) -> dict[str, Any]:
    try:
        spec_path = sb.validate_path(spec_path_raw)
    except Exception:
        return {}
    if not spec_path.exists():
        return {}
    payload = json.loads(spec_path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _required_columns_for_spec(spec_data: dict[str, Any]) -> tuple[str, ...]:
    formula = json.dumps(spec_data, ensure_ascii=False).lower()
    columns = ["symbol", "trade_date", "close"]
    for candidate in ("open", "high", "low", "volume", "amount", "turnover"):
        if candidate in formula and candidate not in columns:
            columns.append(candidate)
    return tuple(columns)


def _requested_symbols(input_data: dict[str, Any]) -> list[str]:
    raw_symbols: list[Any] = []
    symbols_value = input_data.get("symbols", [])
    if isinstance(symbols_value, list):
        raw_symbols.extend(symbols_value)
    elif symbols_value:
        raw_symbols.append(symbols_value)
    for alias in ("symbol", "code", "universe"):
        value = input_data.get(alias)
        if value:
            raw_symbols.append(value)

    normalized: list[str] = []
    for raw in raw_symbols:
        text = str(raw).strip()
        if not text:
            continue
        if "." not in text and text.isdigit() and len(text) == 6:
            text = f"{text}.SZ" if text.startswith(("0", "1", "2", "3")) else f"{text}.SH"
        if text not in normalized:
            normalized.append(text)
    return normalized


def _render_factor_code(name: str, lookback: int, formula: str) -> str:
    safe_name = name.replace(" ", "_").replace("-", "_").lower()
    formula_lower = formula.lower()
    body = _factor_compute_body(safe_name, formula_lower, lookback)
    return f'''"""Candidate factor: {name}.
Auto-generated by Agent. REVIEW_REQUIRED before promotion.
"""

from typing import Any

import pandas as pd


def compute(bars: pd.DataFrame, params: dict[str, Any] | None = None) -> pd.Series:
    """{formula}

    Lookback: {lookback} days.
    """
    if bars.empty:
        return pd.Series(dtype="float64")
    lookback = int((params or {{}}).get("lookback", {lookback}))
{body}
'''


def _factor_compute_body(name: str, formula: str, lookback: int) -> str:
    if "rsi" in name or "relative strength" in formula or "rs =" in formula:
        return '''    delta = bars.groupby("symbol")["close"].diff()
    gain = delta.clip(lower=0)
    loss = (-delta.clip(upper=0)).replace(0, pd.NA)
    avg_gain = gain.groupby(bars["symbol"]).transform(lambda item: item.rolling(14).mean())
    avg_loss = loss.groupby(bars["symbol"]).transform(lambda item: item.rolling(14).mean())
    rs = avg_gain / avg_loss
    return 100 - 100 / (1 + rs)'''
    if "volume_breakout" in name or "volume_ratio" in formula:
        return '''    volume_ma = bars.groupby("symbol")["volume"].transform(
        lambda item: item.rolling(5).mean()
    )
    return bars["volume"] / volume_ma'''
    if "intraday_range" in name or "(high-low)" in formula or "high-low" in formula:
        return '''    daily_range = (bars["high"] - bars["low"]) / bars["close"]
    return daily_range.groupby(bars["symbol"]).transform(lambda item: item.rolling(5).mean())'''
    if "ma_cross" in name or ("ma5" in formula and "ma20" in formula):
        return '''    ma5 = bars.groupby("symbol")["close"].transform(
        lambda item: item.rolling(5).mean()
    )
    ma20 = bars.groupby("symbol")["close"].transform(
        lambda item: item.rolling(20).mean()
    )
    return ma5 / ma20 - 1'''
    return '    return bars.groupby("symbol")["close"].pct_change(lookback)'


def _render_factor_test_code(name: str) -> str:
    return f'''"""Tests for candidate factor: {name}."""

import importlib.util
from pathlib import Path

import pandas as pd


MODULE_PATH = Path(__file__).with_name("factor.py")
SPEC = importlib.util.spec_from_file_location("candidate_factor", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
compute = MODULE.compute


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
