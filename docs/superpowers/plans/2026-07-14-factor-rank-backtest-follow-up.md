# Factor-Rank Backtest Correctness Follow-up Repair Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the remaining correctness gaps in `codex/factor-rank-backtest-correctness` so the factor-rank backtest never silently ignores declared strategy semantics, never hides missing market/universe/accounting evidence, and produces diagnostics from one canonical evidence model.

**Architecture:** Resolve the complete executable strategy contract before any data load, then run a fail-closed pipeline: strategy capability validation → expected trading calendar → point-in-time universe membership → input panel → strict daily ledger → accounting invariants → canonical metrics → diagnostics → governed report. Unsupported user-declared semantics return `BLOCKED`; market-data, universe-timeline, and accounting integrity violations raise typed exceptions and become `ERROR` only at the outer Agent-tool boundary. Unexpected programming exceptions remain uncaught.

**Tech Stack:** Python 3.11+, pandas, Pydantic v2, pytest, existing `DataLake`, `FactorRegistry`, `StrategyRegistry`, `UniverseResolver`, `ArtifactStore`, `ContentAddressedCache`, NiceGUI, Ruff, mypy, and `uv`.

## Branch and Baseline

- Repository: `huangyicong0769/qmt-agent-trader`
- Work branch: `codex/factor-rank-backtest-correctness`
- Merge base: `main@e32586f1f23c2f791fb67e4f8ac01311ce70ac41`
- Current branch state at review: 15 commits ahead, 0 behind.
- Do not restart from `main`.
- Do not squash while implementing. Keep one focused commit per task so each correction can be reviewed independently.
- Save this plan in the repository as:
  `docs/superpowers/plans/2026-07-14-factor-rank-backtest-follow-up.md`

## Review Blockers This Plan Must Close

1. A saved generated strategy with `SavedStrategy.code_path` can still run through the canonical adapter because only the request `code_path` is validated.
2. `cost_drag` and `average_top_n_overlap` diagnostics read the legacy `SensitivityMetrics` payload instead of canonical metrics, producing false values.
3. Missing diagnostic evidence is treated as `PASS` for cost drag and overlap.
4. `execution.cost_model` and non-empty `risk_constraints` are silently ignored.
5. Rolling universe snapshots use first-period dates while strategy signals use last-period dates; exact-key lookup can silently produce no trades.
6. A completely missing market trading session is invisible because expected sessions are inferred from observed bars.
7. Buy affordability excludes minimum commission after quantity reduction, allowing negative cash.

## Global Constraints

- Preserve `research_only=True` and `live_trading_allowed=False`.
- Never execute Agent-generated Python in the main process.
- Never silently replace generated strategy execution with the canonical factor-rank adapter.
- Never silently ignore any non-default `StrategySpec` field.
- Unsupported but valid user intent returns `BLOCKED`, not `ERROR`.
- Market-data, universe-timeline, and accounting integrity failures raise typed exceptions.
- Catch typed integrity exceptions only at the outer Agent-tool boundary.
- Unexpected exceptions must propagate to the normal runtime error handler.
- Do not write a completed report, update the successful-result cache, or calculate partial metrics after an integrity error.
- Do not use previous-close, zero-value, synthetic-bar, empty-universe, or empty-factor fallbacks.
- No new runtime dependency.
- Follow TDD: failing regression → minimal implementation → focused verification → commit.
- Final verification is the focused regression suite plus `make check`.

---

## File Responsibility Map

### Existing files to modify

- `src/qmt_agent_trader/backtest/errors.py`  
  Canonical typed integrity error hierarchy and structured payload fields.

- `src/qmt_agent_trader/backtest/research_runner.py`  
  Strict daily ledger, expected-session validation, rolling-universe as-of lookup, affordability, and accounting invariants.

- `src/qmt_agent_trader/backtest/research_models.py`  
  Completed-run evidence only; remove stale/fallback-era fields.

- `src/qmt_agent_trader/strategy/models.py`  
  Strategy declarations; no new semantics unless needed for explicit validation.

- `src/qmt_agent_trader/strategy/adapter_capabilities.py`  
  Complete capability contract including cost model, risk constraints, and generated implementation paths.

- `src/qmt_agent_trader/strategy/execution_adapter.py`  
  Registry-aware implementation resolution, expected-session loading, canonical metrics before diagnostics, governed report writing.

- `src/qmt_agent_trader/strategy/diagnostics.py`  
  Missing-evidence handling and canonical cost/churn checks.

- `src/qmt_agent_trader/agent/tools/strategy_tools.py`  
  Outer error boundary, effective saved code-path handling, successful-cache rules.

- `src/qmt_agent_trader/universe/resolver.py`  
  Period-end rolling snapshots and initial anchor snapshot.

- `src/qmt_agent_trader/factors/input_panel.py`  
  Retain cross-sectional diagnostics; do not infer complete trading calendars from panel rows.

- `docs/backtest/factor-rank-adapter.md`  
  Updated semantics and failure contract.

### New focused modules

- `src/qmt_agent_trader/data/trading_calendar.py`  
  Load expected open sessions from `raw/tushare/trade_cal`.

- `src/qmt_agent_trader/universe/timeline.py`  
  Point-in-time rolling-universe membership lookup.


---

# Task 1: Establish One Typed Integrity Error Contract

**Files:**
- Modify: `src/qmt_agent_trader/backtest/errors.py`
- Modify: `src/qmt_agent_trader/agent/tools/strategy_tools.py`
- Test: `tests/unit/backtest/test_integrity_errors.py`
- Test: `tests/unit/agent/test_backtest_integrity_error_boundary.py`

**Interfaces:**
- Produces:
  - `BacktestIntegrityError`
  - `BacktestDataIntegrityError`
  - `BacktestUniverseIntegrityError`
  - `BacktestAccountingError`
  - `BacktestIntegrityError.as_dict() -> dict[str, object]`
- The Agent tool catches only `BacktestIntegrityError`.

- [ ] **Step 1: Write the failing error serialization test**

Create `tests/unit/backtest/test_integrity_errors.py`:

```python
from qmt_agent_trader.backtest.errors import BacktestAccountingError


def test_integrity_error_serializes_structured_context() -> None:
    error = BacktestAccountingError(
        code="NEGATIVE_CASH_AFTER_BUY",
        message="post-trade cash violated the non-negative invariant",
        trade_date="2024-01-03",
        symbols=("000001.SZ",),
        field="cash",
        details={"cash": -5.0, "tolerance": 1e-8},
    )

    assert error.as_dict() == {
        "code": "NEGATIVE_CASH_AFTER_BUY",
        "message": "post-trade cash violated the non-negative invariant",
        "trade_date": "2024-01-03",
        "symbols": ["000001.SZ"],
        "field": "cash",
        "details": {"cash": -5.0, "tolerance": 1e-8},
    }
```

- [ ] **Step 2: Run the test and verify failure**

```bash
uv run pytest tests/unit/backtest/test_integrity_errors.py -q
```

Expected: import or constructor failure because the hierarchy does not exist.

- [ ] **Step 3: Replace the single dataclass with a stable hierarchy**

Implement in `src/qmt_agent_trader/backtest/errors.py`:

```python
"""Typed fail-closed errors for research backtests."""

from __future__ import annotations

from typing import Any


class BacktestIntegrityError(RuntimeError):
    def __init__(
        self,
        *,
        code: str,
        message: str,
        trade_date: str | None = None,
        symbols: tuple[str, ...] = (),
        field: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.trade_date = trade_date
        self.symbols = symbols
        self.field = field
        self.details = dict(details or {})

    def as_dict(self) -> dict[str, object]:
        return {
            "code": self.code,
            "message": self.message,
            "trade_date": self.trade_date,
            "symbols": list(self.symbols),
            "field": self.field,
            "details": self.details,
        }

    def __str__(self) -> str:
        return (
            f"{self.code}: {self.message}; trade_date={self.trade_date}; "
            f"field={self.field}; symbols={list(self.symbols)}; details={self.details}"
        )


class BacktestDataIntegrityError(BacktestIntegrityError):
    """Required market or calendar data is absent or invalid."""


class BacktestUniverseIntegrityError(BacktestIntegrityError):
    """Point-in-time universe membership cannot be resolved."""


class BacktestAccountingError(BacktestIntegrityError):
    """The simulated ledger violates an accounting invariant."""
```

This preserves existing keyword construction for `BacktestDataIntegrityError`.

- [ ] **Step 4: Write the outer-boundary regression**

In `tests/unit/agent/test_backtest_integrity_error_boundary.py`:

```python
def test_agent_tool_returns_structured_error_for_known_integrity_failure(
    monkeypatch,
    wired_strategy_tools,
) -> None:
    def fail(*_args, **_kwargs):
        raise BacktestAccountingError(
            code="NEGATIVE_CASH_AFTER_BUY",
            message="cash became negative",
            trade_date="2024-01-03",
            symbols=("000001.SZ",),
            field="cash",
            details={"cash": -5.0},
        )

    monkeypatch.setattr(strategy_tools, "run_strategy_backtest", fail)

    result = strategy_tools._run_backtest(valid_backtest_input(), tool_context())

    assert result["status"] == "ERROR"
    assert result["reason"] == "BACKTEST_INTEGRITY_ERROR"
    assert result["error"]["code"] == "NEGATIVE_CASH_AFTER_BUY"
    assert result["error"]["details"]["cash"] == -5.0
```

Also add:

```python
def test_agent_tool_does_not_swallow_unexpected_runtime_error(
    monkeypatch,
    wired_strategy_tools,
) -> None:
    monkeypatch.setattr(
        strategy_tools,
        "run_strategy_backtest",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("bug")),
    )

    with pytest.raises(RuntimeError, match="bug"):
        strategy_tools._run_backtest(valid_backtest_input(), tool_context())
```

- [ ] **Step 5: Change the tool boundary**

Replace the specific `except BacktestDataIntegrityError` with:

```python
except BacktestIntegrityError as exc:
    return {
        "status": "ERROR",
        "reason": "BACKTEST_INTEGRITY_ERROR",
        "error": exc.as_dict(),
        "research_only": True,
        "live_trading_allowed": False,
    }
```

Do not add `except Exception`.

- [ ] **Step 6: Run focused tests**

```bash
uv run pytest \
  tests/unit/backtest/test_integrity_errors.py \
  tests/unit/agent/test_backtest_integrity_error_boundary.py \
  tests/unit/backtest/test_research_runner_valuation.py -q
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add src/qmt_agent_trader/backtest/errors.py \
        src/qmt_agent_trader/agent/tools/strategy_tools.py \
        tests/unit/backtest/test_integrity_errors.py \
        tests/unit/agent/test_backtest_integrity_error_boundary.py
git commit -m "refactor(backtest): unify fail-closed integrity errors"
```

---

# Task 2: Block Saved Generated Implementations at Every Entry Point

**Files:**
- Modify: `src/qmt_agent_trader/strategy/execution_adapter.py`
- Modify: `src/qmt_agent_trader/agent/tools/strategy_tools.py`
- Modify: `src/qmt_agent_trader/strategy/adapter_capabilities.py`
- Test: `tests/unit/agent/test_saved_generated_strategy_backtest_guard.py`
- Test: `tests/unit/strategy/test_backtest_adapter_capabilities.py`

**Interfaces:**
- Adds `implementation_code_path: str | None` to `StrategyBacktestConfig`.
- Adds `_effective_implementation_code_path(...) -> str | None`.
- `run_strategy_backtest()` validates both request metadata and registry metadata.

- [ ] **Step 1: Write the saved-strategy regression**

Create `tests/unit/agent/test_saved_generated_strategy_backtest_guard.py`:

```python
def test_saved_generated_strategy_is_not_silently_run_by_canonical_adapter(
    wired_strategy_tools,
    saved_generated_strategy,
) -> None:
    result = strategy_tools._run_backtest(
        {
            "strategy_id": saved_generated_strategy.strategy_id,
            "symbols": ["000001.SZ", "000002.SZ"],
            "start_date": "20240101",
            "end_date": "20240331",
        },
        tool_context(),
    )

    assert saved_generated_strategy.code_path
    assert result["status"] == "BLOCKED"
    assert result["reason"] == "GENERATED_STRATEGY_EXECUTION_NOT_IMPLEMENTED"
    assert result["unsupported_fields"] == ["code_path"]
```

Add a positive counterpart:

```python
def test_saved_spec_draft_without_code_path_can_use_canonical_adapter(
    wired_strategy_tools,
    saved_spec_draft,
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        strategy_tools,
        "run_strategy_backtest",
        lambda *_args, **_kwargs: completed_result(),
    )

    result = strategy_tools._run_backtest(
        {
            "strategy_id": saved_spec_draft.strategy_id,
            "symbols": ["000001.SZ", "000002.SZ"],
            "start_date": "20240101",
            "end_date": "20240331",
        },
        tool_context(),
    )

    assert result["status"] == "completed"
```

- [ ] **Step 2: Add the config field**

In `StrategyBacktestConfig`:

```python
implementation_code_path: str | None = None
```

- [ ] **Step 3: Resolve one effective code path in the Agent tool**

Add:

```python
def _effective_implementation_code_path(
    input_data: dict[str, Any],
    saved_strategy: SavedStrategy | None,
) -> str | None:
    explicit = str(input_data.get("code_path") or "").strip()
    if explicit:
        return explicit
    if saved_strategy is None or not saved_strategy.code_path:
        return None
    saved = str(saved_strategy.code_path).strip()
    return saved or None
```

Use it before capability validation:

```python
effective_code_path = _effective_implementation_code_path(input_data, saved_strategy)

capability_issues = validate_factor_rank_adapter_spec(
    strategy_spec,
    code_path=effective_code_path,
)
```

Pass it into `StrategyBacktestConfig`:

```python
implementation_code_path=effective_code_path,
```

- [ ] **Step 4: Make `run_strategy_backtest()` registry-aware**

Replace `_strategy_spec_from_registry()` with:

```python
def _strategy_from_registry(
    registry: StrategyRegistry,
    strategy_id: str,
) -> SavedStrategy | None:
    return registry.get_strategy(strategy_id)
```

At the start of `run_strategy_backtest()`:

```python
saved_strategy = _strategy_from_registry(registry, config.strategy_id)
spec = config.strategy_spec or (saved_strategy.spec if saved_strategy is not None else None)
effective_code_path = (
    config.implementation_code_path
    or (saved_strategy.code_path if saved_strategy is not None else None)
)
```

Validate:

```python
if spec is not None:
    capability_issues = validate_factor_rank_adapter_spec(
        spec,
        code_path=effective_code_path,
    )
    if capability_issues:
        generated_code = any(issue.field == "code_path" for issue in capability_issues)
        return StrategyBacktestResult(
            run_id=run_id,
            strategy_id=config.strategy_id,
            strategy_version=spec.version,
            status="BLOCKED",
            reason=(
                "GENERATED_STRATEGY_EXECUTION_NOT_IMPLEMENTED"
                if generated_code
                else "UNSUPPORTED_STRATEGY_SEMANTICS"
            ),
            unsupported_fields=[issue.field for issue in capability_issues],
            capability_issues=[asdict(issue) for issue in capability_issues],
        )
```

This closes direct adapter callers as well as Agent-tool callers.

- [ ] **Step 5: Remove misleading post-run warnings**

Delete the branch that adds:

```python
"strategy has no generated code; backtest used canonical adapter"
```

Canonical adapter eligibility must be decided before execution, not narrated after execution.

- [ ] **Step 6: Run focused tests**

```bash
uv run pytest \
  tests/unit/agent/test_saved_generated_strategy_backtest_guard.py \
  tests/unit/strategy/test_backtest_adapter_capabilities.py \
  tests/unit/strategy/test_backtest_config_propagation.py -q
```

- [ ] **Step 7: Commit**

```bash
git add src/qmt_agent_trader/strategy/execution_adapter.py \
        src/qmt_agent_trader/agent/tools/strategy_tools.py \
        src/qmt_agent_trader/strategy/adapter_capabilities.py \
        tests/unit/agent/test_saved_generated_strategy_backtest_guard.py \
        tests/unit/strategy/test_backtest_adapter_capabilities.py
git commit -m "fix(strategy): block saved generated code adapter fallback"
```

---

# Task 3: Complete the Adapter Capability Contract

**Files:**
- Modify: `src/qmt_agent_trader/strategy/adapter_capabilities.py`
- Test: `tests/unit/strategy/test_backtest_adapter_capabilities.py`

**Interfaces:**
- Canonical adapter supports:
  - `execution.cost_model == "a_share_default"`
  - `risk_constraints == {}`
- Every non-default unsupported declaration produces one `AdapterCapabilityIssue`.

- [ ] **Step 1: Add failing tests**

```python
@pytest.mark.parametrize(
    ("update", "field"),
    [
        ({"execution": {"cost_model": "zero_cost"}}, "execution.cost_model"),
        ({"risk_constraints": {"stop_loss_pct": 0.10}}, "risk_constraints"),
    ],
)
def test_unimplemented_strategy_semantics_are_reported(update, field) -> None:
    payload = _base_spec().model_dump(mode="json")
    payload = deep_merge(payload, update)

    issues = validate_factor_rank_adapter_spec(StrategySpec.model_validate(payload))

    assert field in {issue.field for issue in issues}
```

Ensure `deep_merge` is used; a shallow `dict.update()` would accidentally discard sibling execution fields.

- [ ] **Step 2: Extend the checks**

Add to `checks`:

```python
(
    "execution.cost_model",
    spec.execution.cost_model,
    "a_share_default",
    spec.execution.cost_model == "a_share_default",
),
```

Add after factor-transform validation:

```python
if spec.risk_constraints:
    issues.append(
        AdapterCapabilityIssue(
            field="risk_constraints",
            observed=dict(spec.risk_constraints),
            supported={},
            message="canonical factor-rank adapter does not execute risk_constraints",
        )
    )
```

- [ ] **Step 3: Add a completeness guard test**

The test should enumerate all current `ExecutionAssumptionSpec` and `PortfolioConstructionSpec` model fields and assert that each is either:

- executed by the adapter and covered by a named test, or
- explicitly validated by `validate_factor_rank_adapter_spec`.

Implement a simple expected set:

```python
def test_capability_contract_tracks_all_declared_strategy_fields() -> None:
    declared = {
        "kind",
        "portfolio.method",
        "portfolio.top_n",
        "portfolio.max_single_position_pct",
        "portfolio.cash_buffer_pct",
        "portfolio.long_only",
        "rebalance.frequency",
        "rebalance.min_turnover_threshold",
        "rebalance.rank_buffer",
        "execution.signal_timing",
        "execution.execution_timing",
        "execution.execution_delay_days",
        "execution.slippage_bps",
        "execution.cost_model",
        "factors[].ascending",
        "factors[].weight",
        "factors[].transform",
        "risk_constraints",
    }
    assert CANONICAL_FACTOR_RANK_SEMANTIC_FIELDS == declared
```

Define `CANONICAL_FACTOR_RANK_SEMANTIC_FIELDS` beside the validator. This makes future model additions fail tests until capability handling is decided.

- [ ] **Step 4: Run tests**

```bash
uv run pytest tests/unit/strategy/test_backtest_adapter_capabilities.py -q
```

- [ ] **Step 5: Commit**

```bash
git add src/qmt_agent_trader/strategy/adapter_capabilities.py \
        tests/unit/strategy/test_backtest_adapter_capabilities.py
git commit -m "fix(strategy): validate complete adapter semantics"
```

---

# Task 4: Build Canonical Metrics Before Diagnostics

**Files:**
- Modify: `src/qmt_agent_trader/strategy/execution_adapter.py`
- Modify: `src/qmt_agent_trader/strategy/diagnostics.py`
- Modify: `src/qmt_agent_trader/backtest/research_models.py`
- Test: `tests/unit/strategy/test_backtest_diagnostic_wiring.py`
- Modify: `tests/unit/backtest/test_research_models.py`

**Interfaces:**
- Produces `_build_canonical_metrics(result, config) -> dict[str, object]`.
- `_diagnostic_evidence(..., canonical_metrics: Mapping[str, object])`.
- Missing cost/churn evidence becomes `NOT_COMPUTED`.

- [ ] **Step 1: Write the diagnostic wiring regression**

Create `tests/unit/strategy/test_backtest_diagnostic_wiring.py`:

```python
def test_diagnostics_use_canonical_cost_drag_and_overlap() -> None:
    result = factor_rank_result(
        net_return=-0.20,
        same_trade_gross_return=0.05,
        average_top_n_overlap=0.80,
        explicit_cost=10_000.0,
        slippage_cost=5_000.0,
    )
    config = strategy_backtest_config(initial_cash=1_000_000)

    metrics = execution_adapter._build_canonical_metrics(result, config)
    evidence = execution_adapter._diagnostic_evidence(
        result.as_dict(),
        {"valid": True, "execution_delay_days": 1},
        canonical_metrics=metrics,
        factor_frame=factor_frame(),
        bars=bars(),
        initial_cash=config.initial_cash,
    )
    diagnostics = StrategyDiagnosticsEvaluator().evaluate(evidence)

    checks = {check.name: check for check in diagnostics.checks}
    assert checks["cost_drag"].observed == 0.25
    assert checks["average_top_n_overlap"].observed == 0.80
```

- [ ] **Step 2: Write missing-evidence regressions**

```python
def test_missing_cost_drag_is_not_computed_not_passed() -> None:
    diagnostics = StrategyDiagnosticsEvaluator().evaluate(minimal_evidence())
    checks = {check.name: check for check in diagnostics.checks}
    assert checks["cost_drag"].status == DiagnosticStatus.NOT_COMPUTED


def test_missing_overlap_is_not_computed_not_passed() -> None:
    diagnostics = StrategyDiagnosticsEvaluator().evaluate(minimal_evidence())
    checks = {check.name: check for check in diagnostics.checks}
    assert checks["average_top_n_overlap"].status == DiagnosticStatus.NOT_COMPUTED
```

- [ ] **Step 3: Extract canonical metric construction**

Add:

```python
def _build_canonical_metrics(
    result: FactorRankResearchResult,
    config: StrategyBacktestConfig,
) -> dict[str, object]:
    net_return = result.metrics.total_return
    gross_return = result.same_trade_gross_return
    return {
        "total_return": round(net_return, 6),
        "net_total_return": round(net_return, 6),
        "same_trade_gross_return": round(gross_return, 6),
        "cost_drag": round(gross_return - net_return, 6),
        "sharpe": round(result.metrics.sharpe, 6),
        "max_drawdown": round(result.metrics.max_drawdown, 6),
        "turnover": round(result.metrics.turnover, 6),
        "average_one_way_turnover": round(result.metrics.turnover, 6),
        "average_top_n_overlap": round(result.average_top_n_overlap, 6),
        "explicit_cost_to_initial_cash": round(
            result.total_explicit_cost / config.initial_cash,
            6,
        ),
        "slippage_cost_to_initial_cash": round(
            result.total_slippage_cost / config.initial_cash,
            6,
        ),
        "trade_count": len(result.trades),
    }
```

Call it immediately after `runner.run()` and before `_diagnostic_evidence()`.

- [ ] **Step 4: Pass canonical metrics explicitly**

Change the signature:

```python
def _diagnostic_evidence(
    result_dict: dict[str, Any],
    leakage_report: dict[str, object],
    *,
    canonical_metrics: Mapping[str, object],
    factor_frame: pd.DataFrame,
    bars: pd.DataFrame,
    initial_cash: float,
) -> dict[str, Any]:
```

Use:

```python
"turnover_report": {
    "average_turnover": canonical_metrics["average_one_way_turnover"],
},
"cost_report": {
    "cost_to_initial_cash": (
        float(canonical_metrics["explicit_cost_to_initial_cash"])
        + float(canonical_metrics["slippage_cost_to_initial_cash"])
    ),
    "cost_drag": canonical_metrics["cost_drag"],
},
"churn_report": {
    "average_top_n_overlap": canonical_metrics["average_top_n_overlap"],
},
```

- [ ] **Step 5: Make missing evidence explicit**

Replace the current fallback `PASS` returns in `_top_n_overlap_check()` and `_cost_drag_check()` with `_not_computed_check(...)`.

- [ ] **Step 6: Remove fallback-era result fields**

In `ResearchDataQuality`, remove:

```python
missing_held_price_events
stale_valuation_dates
```

A completed strict run cannot contain these. Integrity failures are exceptions, not completed-run counters.

Update serialization tests accordingly.

- [ ] **Step 7: Run focused tests**

```bash
uv run pytest \
  tests/unit/strategy/test_backtest_diagnostic_wiring.py \
  tests/unit/test_strategy_diagnostics.py \
  tests/unit/backtest/test_research_models.py \
  tests/unit/backtest/test_research_runner_attribution.py -q
```

- [ ] **Step 8: Commit**

```bash
git add src/qmt_agent_trader/strategy/execution_adapter.py \
        src/qmt_agent_trader/strategy/diagnostics.py \
        src/qmt_agent_trader/backtest/research_models.py \
        tests/unit/strategy/test_backtest_diagnostic_wiring.py \
        tests/unit/test_strategy_diagnostics.py \
        tests/unit/backtest/test_research_models.py
git commit -m "fix(backtest): wire diagnostics to canonical metrics"
```

---

# Task 5: Implement Point-in-Time Rolling Universe Membership

**Files:**
- Create: `src/qmt_agent_trader/universe/timeline.py`
- Modify: `src/qmt_agent_trader/universe/resolver.py`
- Modify: `src/qmt_agent_trader/backtest/research_runner.py`
- Create: `tests/unit/universe/test_timeline.py`
- Modify: `tests/unit/universe/test_resolver.py`
- Create: `tests/unit/backtest/test_research_runner_rolling_universe.py`

**Interfaces:**
- Produces:
  - `RollingUniverseTimeline.from_mapping(...)`
  - `RollingUniverseTimeline.membership_as_of(as_of_date) -> tuple[str, ...]`
- Missing initial membership raises `BacktestUniverseIntegrityError`.

- [ ] **Step 1: Write timeline tests**

Create `tests/unit/universe/test_timeline.py`:

```python
from datetime import date

import pytest

from qmt_agent_trader.backtest.errors import BacktestUniverseIntegrityError
from qmt_agent_trader.universe.timeline import RollingUniverseTimeline


def test_membership_uses_latest_snapshot_on_or_before_signal_date() -> None:
    timeline = RollingUniverseTimeline.from_mapping(
        {
            "20240105": ["A", "B"],
            "20240112": ["B", "C"],
        }
    )

    assert timeline.membership_as_of(date(2024, 1, 8)) == ("A", "B")
    assert timeline.membership_as_of(date(2024, 1, 12)) == ("B", "C")


def test_membership_before_first_snapshot_raises() -> None:
    timeline = RollingUniverseTimeline.from_mapping({"20240105": ["A"]})

    with pytest.raises(BacktestUniverseIntegrityError) as exc_info:
        timeline.membership_as_of(date(2024, 1, 4))

    assert exc_info.value.code == "ROLLING_UNIVERSE_SNAPSHOT_NOT_AVAILABLE"
```

- [ ] **Step 2: Implement the timeline**

```python
"""Point-in-time rolling universe membership."""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
from datetime import date, datetime
from typing import Mapping, Sequence

from qmt_agent_trader.backtest.errors import BacktestUniverseIntegrityError


def _parse_date_key(value: str) -> date:
    for fmt in ("%Y%m%d", "%Y-%m-%d"):
        try:
            return datetime.strptime(value, fmt).date()
        except ValueError:
            continue
    raise ValueError(f"invalid rolling universe date key: {value}")


@dataclass(frozen=True)
class RollingUniverseTimeline:
    dates: tuple[date, ...]
    membership_by_date: dict[date, tuple[str, ...]]

    @classmethod
    def from_mapping(
        cls,
        mapping: Mapping[str, Sequence[str]],
    ) -> "RollingUniverseTimeline":
        normalized = {
            _parse_date_key(str(key)): tuple(dict.fromkeys(str(item) for item in symbols))
            for key, symbols in mapping.items()
        }
        return cls(
            dates=tuple(sorted(normalized)),
            membership_by_date=normalized,
        )

    def membership_as_of(self, as_of_date: date) -> tuple[str, ...]:
        index = bisect_right(self.dates, as_of_date) - 1
        if index < 0:
            raise BacktestUniverseIntegrityError(
                code="ROLLING_UNIVERSE_SNAPSHOT_NOT_AVAILABLE",
                message="no rolling-universe snapshot exists on or before signal date",
                trade_date=f"{as_of_date:%Y-%m-%d}",
                field="symbols_by_date",
                details={
                    "first_available_snapshot": (
                        f"{self.dates[0]:%Y-%m-%d}" if self.dates else None
                    )
                },
            )
        return self.membership_by_date[self.dates[index]]
```

- [ ] **Step 3: Make rolling resolver dates period-end with an initial anchor**

Replace first-seen bucket logic with period-end logic. The rolling date set must contain:

1. The first available trading date in the requested range as an anchor.
2. The final available trading date of each week/month.

Implement:

```python
def _period_end_dates(dates: list[str], frequency: str) -> list[str]:
    if not dates:
        return []
    if frequency == "daily":
        return dates
    last_by_bucket: dict[tuple[int, int], str] = {}
    for key in dates:
        parsed = _parse_date(key)
        if frequency == "weekly":
            iso = parsed.isocalendar()
            bucket = (iso.year, iso.week)
        elif frequency == "monthly":
            bucket = (parsed.year, parsed.month)
        else:
            raise ValueError(f"unsupported rebalance frequency: {frequency}")
        last_by_bucket[bucket] = key
    selected = [dates[0], *last_by_bucket.values()]
    return list(dict.fromkeys(selected))
```

Use this from `_rebalance_dates()`.

- [ ] **Step 4: Use as-of membership in the runner**

Construct once:

```python
self._universe_timeline = (
    RollingUniverseTimeline.from_mapping(config.symbols_by_date)
    if config.symbols_by_date
    else None
)
```

Replace exact dictionary lookup:

```python
if self._universe_timeline is None:
    return filtered
symbols = self._universe_timeline.membership_as_of(signal_date)
return filtered[filtered["symbol"].astype(str).isin(symbols)].copy()
```

If resolved membership is empty, raise:

```python
BacktestUniverseIntegrityError(
    code="ROLLING_UNIVERSE_EMPTY_AS_OF_SIGNAL",
    message="resolved rolling-universe membership is empty",
    trade_date=f"{signal_date:%Y-%m-%d}",
    field="symbols_by_date",
)
```

- [ ] **Step 5: Add strategy/universe frequency matrix tests**

In `tests/unit/backtest/test_research_runner_rolling_universe.py`, cover:

- daily strategy + weekly snapshots;
- weekly strategy + weekly snapshots;
- monthly strategy + monthly snapshots;
- missing initial snapshot;
- empty resolved snapshot.

Assert that valid combinations execute trades instead of silently producing zero trades.

- [ ] **Step 6: Run focused tests**

```bash
uv run pytest \
  tests/unit/universe/test_timeline.py \
  tests/unit/universe/test_resolver.py \
  tests/unit/backtest/test_research_runner_rolling_universe.py -q
```

- [ ] **Step 7: Commit**

```bash
git add src/qmt_agent_trader/universe/timeline.py \
        src/qmt_agent_trader/universe/resolver.py \
        src/qmt_agent_trader/backtest/research_runner.py \
        tests/unit/universe/test_timeline.py \
        tests/unit/universe/test_resolver.py \
        tests/unit/backtest/test_research_runner_rolling_universe.py
git commit -m "fix(universe): resolve rolling membership point in time"
```

---

# Task 6: Validate Against an Independent Trading Calendar

**Files:**
- Create: `src/qmt_agent_trader/data/trading_calendar.py`
- Modify: `src/qmt_agent_trader/strategy/execution_adapter.py`
- Modify: `src/qmt_agent_trader/backtest/research_runner.py`
- Create: `tests/unit/data/test_trading_calendar.py`
- Create: `tests/unit/backtest/test_expected_sessions.py`
- Modify: all direct `FactorRankResearchConfig(...)` test helpers to pass expected dates.

**Interfaces:**
- Produces:
  - `load_open_sessions(lake, start, end, exchanges=("SSE", "SZSE")) -> tuple[date, ...]`
  - Required `expected_trade_dates: tuple[date, ...]` in `FactorRankResearchConfig`.
- Missing or mismatched sessions raise `BacktestDataIntegrityError`.

- [ ] **Step 1: Write calendar-loader tests**

Create `tests/unit/data/test_trading_calendar.py`:

```python
def test_load_open_sessions_uses_trade_cal_not_observed_bars(tmp_path) -> None:
    lake = data_lake(tmp_path)
    write_trade_cal(
        lake,
        [
            {"exchange": "SSE", "cal_date": "20240102", "is_open": 1},
            {"exchange": "SSE", "cal_date": "20240103", "is_open": 1},
            {"exchange": "SSE", "cal_date": "20240104", "is_open": 0},
        ],
    )

    assert load_open_sessions(lake, start="20240102", end="20240104") == (
        date(2024, 1, 2),
        date(2024, 1, 3),
    )
```

Add:

```python
def test_missing_trade_calendar_raises() -> None:
    with pytest.raises(BacktestDataIntegrityError) as exc_info:
        load_open_sessions(empty_lake(), start="20240102", end="20240104")
    assert exc_info.value.code == "TRADING_CALENDAR_NOT_READY"
```

- [ ] **Step 2: Implement the loader**

```python
"""Canonical expected trading sessions for backtest integrity checks."""

from __future__ import annotations

from datetime import date

import pandas as pd

from qmt_agent_trader.backtest.errors import BacktestDataIntegrityError
from qmt_agent_trader.data.storage import DataLake


def load_open_sessions(
    lake: DataLake,
    *,
    start: str,
    end: str,
    exchanges: tuple[str, ...] = ("SSE", "SZSE"),
) -> tuple[date, ...]:
    dataset = "tushare/trade_cal"
    if not lake.dataset_path("raw", dataset).exists():
        raise BacktestDataIntegrityError(
            code="TRADING_CALENDAR_NOT_READY",
            message="raw/tushare/trade_cal is required for backtest session validation",
            field="trade_cal",
            details={"start": start, "end": end, "exchanges": list(exchanges)},
        )
    frame = lake.read_parquet_filtered(
        "raw",
        dataset,
        columns=["exchange", "cal_date", "is_open"],
        start=start,
        end=end,
        date_column="cal_date",
    )
    required = {"cal_date", "is_open"}
    missing = required.difference(frame.columns)
    if missing:
        raise BacktestDataIntegrityError(
            code="TRADING_CALENDAR_INVALID",
            message="trade calendar is missing required columns",
            field="trade_cal",
            details={"missing_columns": sorted(missing)},
        )
    if "exchange" in frame.columns:
        frame = frame[frame["exchange"].astype(str).isin(exchanges)]
    frame = frame[pd.to_numeric(frame["is_open"], errors="coerce") == 1]
    dates = tuple(sorted(pd.to_datetime(frame["cal_date"]).dt.date.unique()))
    if not dates:
        raise BacktestDataIntegrityError(
            code="TRADING_CALENDAR_EMPTY",
            message="trade calendar contains no open sessions for requested range",
            field="trade_cal",
            details={"start": start, "end": end},
        )
    return dates
```

- [ ] **Step 3: Make expected sessions required by the runner**

Change `FactorRankResearchConfig`:

```python
factor_name: str
expected_trade_dates: tuple[date, ...]
```

Do not give it a default.

Before factor computation:

```python
observed_dates = set(self.bars["trade_date"])
expected_dates = set(config.expected_trade_dates)
missing_dates = sorted(expected_dates - observed_dates)
unexpected_dates = sorted(observed_dates - expected_dates)

if missing_dates:
    raise BacktestDataIntegrityError(
        code="MISSING_EXPECTED_TRADING_SESSION",
        message="one or more expected open sessions have no market bars",
        field="trade_date",
        details={"missing_dates": [f"{item:%Y-%m-%d}" for item in missing_dates]},
    )
if unexpected_dates:
    raise BacktestDataIntegrityError(
        code="UNEXPECTED_MARKET_SESSION",
        message="market bars contain dates not marked open by the trading calendar",
        field="trade_date",
        details={"unexpected_dates": [f"{item:%Y-%m-%d}" for item in unexpected_dates]},
    )
```

Use `config.expected_trade_dates` as the canonical `dates` sequence in `run()`.

- [ ] **Step 4: Load sessions in the execution adapter**

Before constructing the runner:

```python
expected_trade_dates = load_open_sessions(
    lake,
    start=config.start_date,
    end=config.end_date,
)
```

Pass them into `FactorRankResearchConfig`.

- [ ] **Step 5: Write the full-day-gap regression**

Create `tests/unit/backtest/test_expected_sessions.py`:

```python
def test_completely_missing_open_session_aborts_before_execution() -> None:
    expected = (
        date(2024, 1, 2),
        date(2024, 1, 3),
        date(2024, 1, 4),
    )
    bars = bars_for_dates((expected[0], expected[2]))

    with pytest.raises(BacktestDataIntegrityError) as exc_info:
        FactorRankResearchRunner(
            bars,
            FactorRankResearchConfig(
                factor_name="fixture",
                expected_trade_dates=expected,
            ),
        )

    assert exc_info.value.code == "MISSING_EXPECTED_TRADING_SESSION"
    assert exc_info.value.details["missing_dates"] == ["2024-01-03"]
```

- [ ] **Step 6: Update direct runner tests**

Every `FactorRankResearchConfig(...)` in unit/integration tests must pass the exact date tuple from its bars fixture:

```python
expected_trade_dates=tuple(sorted(bars["trade_date"].unique())),
```

Do not add a production fallback that infers expected dates from observed bars.

- [ ] **Step 7: Run focused tests**

```bash
uv run pytest \
  tests/unit/data/test_trading_calendar.py \
  tests/unit/backtest/test_expected_sessions.py \
  tests/unit/backtest \
  tests/unit/strategy/test_backtest_daily_coverage.py -q
```

- [ ] **Step 8: Commit**

```bash
git add src/qmt_agent_trader/data/trading_calendar.py \
        src/qmt_agent_trader/strategy/execution_adapter.py \
        src/qmt_agent_trader/backtest/research_runner.py \
        tests/unit/data/test_trading_calendar.py \
        tests/unit/backtest/test_expected_sessions.py \
        tests/unit/backtest \
        tests/integration/test_factor_rank_backtest_correctness.py
git commit -m "fix(backtest): validate complete exchange sessions"
```

---

# Task 7: Enforce Buy Affordability and Ledger Invariants

**Files:**
- Modify: `src/qmt_agent_trader/backtest/research_runner.py`
- Create: `tests/unit/backtest/test_accounting_invariants.py`

**Interfaces:**
- Produces:
  - `_max_affordable_buy_quantity(...) -> int`
  - `_assert_ledger_invariants(...) -> None`
- Violations raise `BacktestAccountingError`.

- [ ] **Step 1: Write the minimum-commission regression**

Create `tests/unit/backtest/test_accounting_invariants.py`:

```python
def test_buy_affordability_includes_minimum_commission() -> None:
    quantity = _max_affordable_buy_quantity(
        cash=1_000.0,
        price=10.0,
        desired_quantity=100,
        cost_config=CostConfig(min_commission=5.0),
    )

    assert quantity == 0
```

Add:

```python
def test_post_trade_negative_cash_raises_accounting_error() -> None:
    with pytest.raises(BacktestAccountingError) as exc_info:
        _assert_ledger_invariants(
            cash=-0.01,
            positions={"000001.SZ": 100},
            trade_date=date(2024, 1, 3),
        )
    assert exc_info.value.code == "NEGATIVE_CASH_AFTER_TRADE"
```

- [ ] **Step 2: Implement lot-aware affordability**

```python
_CASH_EPSILON = 1e-8


def _max_affordable_buy_quantity(
    *,
    cash: float,
    price: float,
    desired_quantity: int,
    cost_config: CostConfig,
) -> int:
    desired_lots = max(0, desired_quantity // 100)
    low = 0
    high = desired_lots
    while low < high:
        middle = (low + high + 1) // 2
        quantity = middle * 100
        notional = quantity * price
        total = notional + calculate_cost_breakdown(
            notional,
            Side.BUY,
            cost_config,
        ).total
        if total <= cash + _CASH_EPSILON:
            low = middle
        else:
            high = middle - 1
    return low * 100
```

Replace the current `cash / price` estimate with this helper.

- [ ] **Step 3: Implement ledger invariants**

```python
def _assert_ledger_invariants(
    *,
    cash: float,
    positions: dict[str, int],
    trade_date: date,
) -> None:
    if not math.isfinite(cash):
        raise BacktestAccountingError(
            code="NON_FINITE_CASH",
            message="cash must remain finite",
            trade_date=f"{trade_date:%Y-%m-%d}",
            field="cash",
            details={"cash": cash},
        )
    if cash < -_CASH_EPSILON:
        raise BacktestAccountingError(
            code="NEGATIVE_CASH_AFTER_TRADE",
            message="post-trade cash violated the non-negative invariant",
            trade_date=f"{trade_date:%Y-%m-%d}",
            field="cash",
            details={"cash": cash, "tolerance": _CASH_EPSILON},
        )
    invalid_positions = {
        symbol: quantity
        for symbol, quantity in positions.items()
        if quantity <= 0
    }
    if invalid_positions:
        raise BacktestAccountingError(
            code="INVALID_POSITION_QUANTITY",
            message="completed ledger positions must have positive quantities",
            trade_date=f"{trade_date:%Y-%m-%d}",
            field="positions",
            symbols=tuple(sorted(invalid_positions)),
            details={"positions": invalid_positions},
        )
```

Call after every applied trade and once at end of day.

- [ ] **Step 4: Add equity invariants**

After EOD valuation:

```python
if not math.isfinite(equity_after) or equity_after < -_CASH_EPSILON:
    raise BacktestAccountingError(
        code="INVALID_EQUITY_VALUE",
        message="daily equity must be finite and non-negative",
        trade_date=f"{trade_date:%Y-%m-%d}",
        field="equity",
        details={"equity": equity_after},
    )
```

- [ ] **Step 5: Run focused tests**

```bash
uv run pytest \
  tests/unit/backtest/test_accounting_invariants.py \
  tests/unit/backtest/test_research_runner_rebalance.py \
  tests/unit/backtest/test_research_runner_attribution.py -q
```

- [ ] **Step 6: Update the adapter documentation**

Modify `docs/backtest/factor-rank-adapter.md` so it states:

1. Saved or explicit generated code paths block canonical execution.
2. Only `a_share_default` is supported by the canonical adapter.
3. Non-empty `risk_constraints` block execution.
4. Rolling membership uses the latest snapshot on or before the signal date.
5. Rolling snapshots include an initial anchor and period-end updates.
6. Expected sessions come from `raw/tushare/trade_cal`.
7. Missing full sessions raise `MISSING_EXPECTED_TRADING_SESSION`.
8. Missing symbol bars or invalid prices raise typed data-integrity errors.
9. Negative cash or invalid ledger values raise typed accounting errors.
10. Missing diagnostic evidence is `NOT_COMPUTED`, never `PASS`.
11. Integrity failures create no completed report and are not stored in the successful-result cache.

- [ ] **Step 7: Run the complete local verification gate**

```bash
uv run pytest \
  tests/unit/backtest \
  tests/unit/strategy \
  tests/unit/universe \
  tests/unit/data/test_trading_calendar.py \
  tests/unit/agent/test_backtest_integrity_error_boundary.py \
  tests/unit/agent/test_saved_generated_strategy_backtest_guard.py -q
make check
```

Expected: all commands exit 0.

- [ ] **Step 8: Verify broad exception swallowing was not reintroduced**

```bash
rg -n "except Exception" \
  src/qmt_agent_trader/backtest \
  src/qmt_agent_trader/strategy/execution_adapter.py \
  src/qmt_agent_trader/agent/tools/strategy_tools.py
```

Review every result. Broad catches are not allowed in the runner or adapter execution path. Artifact-enumeration/UI catches may remain only when they log and exclude unreadable artifacts.

- [ ] **Step 9: Commit**

```bash
git add src/qmt_agent_trader/backtest/research_runner.py \
        tests/unit/backtest/test_accounting_invariants.py \
        docs/backtest/factor-rank-adapter.md
git commit -m "fix(backtest): enforce cash and ledger invariants"
```

---

---

# Final Review Checklist

Do not request merge until every statement is true.

## Execution semantics

- [ ] A saved strategy with `code_path` cannot run through the canonical adapter.
- [ ] An explicit request `code_path` cannot run through the canonical adapter.
- [ ] A spec-only draft with no code path may use the canonical adapter.
- [ ] Unsupported `cost_model` returns `BLOCKED`.
- [ ] Non-empty `risk_constraints` return `BLOCKED`.
- [ ] Every declared strategy semantic field is tracked by the capability completeness test.

## Integrity behavior

- [ ] Missing symbol-day bars raise typed data errors.
- [ ] Invalid open/close prices raise typed data errors.
- [ ] A completely missing expected exchange session raises `MISSING_EXPECTED_TRADING_SESSION`.
- [ ] A missing rolling-universe snapshot raises a typed universe error.
- [ ] Empty rolling membership raises a typed universe error.
- [ ] Negative cash raises a typed accounting error.
- [ ] Non-finite cash or equity raises a typed accounting error.
- [ ] No integrity error creates a completed report.
- [ ] No integrity error enters the successful backtest cache.
- [ ] Unexpected software exceptions propagate.

## Evidence and diagnostics

- [ ] Canonical metrics are built once before diagnostics.
- [ ] Diagnostics read canonical `cost_drag`.
- [ ] Diagnostics read canonical `average_top_n_overlap`.
- [ ] Missing cost/churn evidence is `NOT_COMPUTED`.
- [ ] Completed-run data quality contains no stale-price fallback fields.
- [ ] Schema v2 report metrics equal returned result metrics.

## Universe and timeline

- [ ] Rolling universe lookup is as-of, not exact-key.
- [ ] Weekly/monthly resolver snapshots are period-end.
- [ ] The first in-range trading date is an anchor snapshot.
- [ ] Daily strategy + weekly universe is covered by focused tests.
- [ ] Weekly strategy + weekly universe is covered by focused tests.
- [ ] Monthly strategy + monthly universe is covered by focused tests.

## Local verification

- [ ] All new focused unit and component tests pass.
- [ ] `make check` passes.
- [ ] No broad exception handler swallows runner or adapter execution failures.
- [ ] Documentation matches the implemented fail-closed semantics.
- [ ] The branch remains `research_only` and cannot authorize live trading.

## Out of Scope for This Plan

The following are intentionally excluded:

- Reproducing the historical approximately `-98%` run, because the Codex environment does not contain the original run artifact or session record.
- Building a new full DataLake-to-Agent end-to-end fixture solely for this follow-up.
- Adding or modifying GitHub Actions; the repository owner will configure CI manually.

## Expected Merge Decision

Only after all checklist items and local verification pass should the branch be reconsidered for approval. Until then the correct review status remains `REQUEST_CHANGES`.

