# Factor-Rank Backtest Final Correctness Repairs Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the remaining correctness gaps on `codex/factor-rank-backtest-correctness` so strategy semantics have one authoritative execution meaning, ambiguous inputs fail closed, and completed reports never invent evidence.

**Architecture:** `StrategySpec` is authoritative whenever it exists. `StrategyBacktestConfig` remains a runtime transport object, but conflicting semantic fields block before cache lookup or data loading. Calendar, universe order, symbol-date uniqueness, signal availability, and numeric constraints are validated explicitly. Missing evidence is represented as `None` and diagnosed as `NOT_COMPUTED`.

**Tech Stack:** Python 3.11+, pandas, Pydantic v2, dataclasses, pytest, existing DataLake/StrategyRegistry/UniverseResolver/factor-rank runner, Ruff, mypy, and uv.

## Global Constraints

- Target branch: `codex/factor-rank-backtest-correctness`; continue from its current head.
- Save this plan as `docs/superpowers/plans/2026-07-14-factor-rank-backtest-final-corrections.md`.
- Keep one focused commit per task.
- `StrategySpec` is authoritative whenever present.
- Ordinary backtest config must not silently override strategy semantics.
- Unsupported intent returns `BLOCKED`.
- Data, universe, and accounting integrity violations raise typed `BacktestIntegrityError` subclasses.
- Only the outer Agent-tool boundary converts typed integrity errors to structured `ERROR`.
- Unexpected programming exceptions propagate.
- Never use stale-price, zero-price, synthetic-bar, first-duplicate-row, empty-universe, or empty-factor fallbacks.
- Never write a completed report or successful cache entry after an integrity failure.
- Missing evidence is `None`, never a fabricated `0.0`.
- Preserve `research_only=True` and `live_trading_allowed=False`.
- No new runtime dependency.
- TDD for every task: failing regression, minimal implementation, focused tests, commit.
- Local verification only. Do not add GitHub Actions.
- Do not recreate the historical extreme-drawdown run.
- Do not add a new full DataLake-to-Agent fixture solely for this follow-up.

---

## Task 1: Make StrategySpec the Single Execution Authority

**Files**
- Modify: `src/qmt_agent_trader/strategy/execution_adapter.py`
- Modify: `src/qmt_agent_trader/agent/tools/strategy_tools.py`
- Create: `tests/unit/strategy/test_backtest_config_spec_consistency.py`
- Create: `tests/unit/agent/test_backtest_config_spec_consistency.py`

**Interface**
- Add `validate_backtest_config_matches_spec(config, spec) -> tuple[AdapterCapabilityIssue, ...]`.
- Mismatches return `BLOCKED` with reason `CONFIG_SPEC_MISMATCH`.
- The check runs before cache lookup, data loading, report writing, and factor computation.

- [ ] **Step 1: Write the direct validator test**

```python
import pytest

from qmt_agent_trader.strategy.execution_adapter import (
    StrategyBacktestConfig,
    validate_backtest_config_matches_spec,
)
from qmt_agent_trader.strategy.models import StrategySpec


def spec() -> StrategySpec:
    return StrategySpec.model_validate({
        "strategy_id": "weekly_value",
        "name": "Weekly value",
        "kind": "FACTOR_RANK_LONG_ONLY",
        "factors": [{"factor_id": "pb", "ascending": True}],
        "portfolio": {
            "method": "equal_weight_top_n",
            "top_n": 10,
            "max_single_position_pct": 0.08,
            "cash_buffer_pct": 0.10,
            "long_only": True,
        },
        "rebalance": {
            "frequency": "weekly",
            "min_turnover_threshold": 0.05,
            "rank_buffer": 10,
        },
        "execution": {
            "signal_timing": "after_close",
            "execution_timing": "next_open",
            "execution_delay_days": 1,
            "slippage_bps": 5.0,
            "cost_model": "a_share_default",
        },
    })


def matching_config() -> StrategyBacktestConfig:
    return StrategyBacktestConfig(
        strategy_id="weekly_value",
        strategy_spec=spec(),
        factor_name="pb",
        start_date="20240101",
        end_date="20240630",
        top_n=10,
        max_single_position_pct=0.08,
        cash_buffer_pct=0.10,
        rebalance_frequency="weekly",
        min_turnover_threshold=0.05,
        rank_buffer=10,
        execution_delay_days=1,
        slippage_bps=5.0,
        lower_is_better=True,
    )


@pytest.mark.parametrize(
    ("update", "field"),
    [
        ({"factor_name": "momentum_20d"}, "config.factor_name"),
        ({"top_n": 20}, "config.top_n"),
        ({"max_single_position_pct": 0.10}, "config.max_single_position_pct"),
        ({"cash_buffer_pct": 0.02}, "config.cash_buffer_pct"),
        ({"rebalance_frequency": "daily"}, "config.rebalance_frequency"),
        ({"min_turnover_threshold": 0.0}, "config.min_turnover_threshold"),
        ({"rank_buffer": 0}, "config.rank_buffer"),
        ({"execution_delay_days": 2}, "config.execution_delay_days"),
        ({"slippage_bps": 10.0}, "config.slippage_bps"),
        ({"lower_is_better": False}, "config.lower_is_better"),
    ],
)
def test_config_cannot_override_spec(update, field):
    config = matching_config().model_copy(update=update)
    issues = validate_backtest_config_matches_spec(config, spec())
    assert field in {issue.field for issue in issues}


def test_matching_config_has_no_mismatch():
    assert validate_backtest_config_matches_spec(matching_config(), spec()) == ()
```

- [ ] **Step 2: Run and confirm failure**

```bash
uv run pytest tests/unit/strategy/test_backtest_config_spec_consistency.py -q
```

- [ ] **Step 3: Implement the validator**

In `execution_adapter.py`:

```python
import math

from qmt_agent_trader.strategy.adapter_capabilities import (
    AdapterCapabilityIssue,
    validate_factor_rank_adapter_spec,
)


def _same_semantic_value(observed: object, expected: object) -> bool:
    if isinstance(observed, float) or isinstance(expected, float):
        try:
            return math.isclose(float(observed), float(expected), rel_tol=1e-12, abs_tol=1e-12)
        except (TypeError, ValueError):
            return False
    return observed == expected


def validate_backtest_config_matches_spec(
    config: StrategyBacktestConfig,
    spec: StrategySpec,
) -> tuple[AdapterCapabilityIssue, ...]:
    first = spec.factors[0] if spec.factors else None
    expected_factor = first.factor_id if first is not None else None
    expected_direction = bool(len(spec.factors) == 1 and first and first.ascending)
    checks = (
        ("config.factor_name", config.factor_name, expected_factor),
        ("config.top_n", config.top_n, spec.portfolio.top_n),
        ("config.max_single_position_pct", config.max_single_position_pct,
         spec.portfolio.max_single_position_pct),
        ("config.cash_buffer_pct", config.cash_buffer_pct, spec.portfolio.cash_buffer_pct),
        ("config.rebalance_frequency", config.rebalance_frequency,
         spec.rebalance.frequency),
        ("config.min_turnover_threshold", config.min_turnover_threshold,
         spec.rebalance.min_turnover_threshold),
        ("config.rank_buffer", config.rank_buffer, spec.rebalance.rank_buffer),
        ("config.execution_delay_days", config.execution_delay_days,
         spec.execution.execution_delay_days),
        ("config.slippage_bps", config.slippage_bps, spec.execution.slippage_bps),
        ("config.lower_is_better", config.lower_is_better, expected_direction),
    )
    return tuple(
        AdapterCapabilityIssue(
            field=field,
            observed=observed,
            supported=expected,
            message=(
                "backtest config conflicts with authoritative StrategySpec: "
                f"{field}={observed!r}, expected {expected!r}"
            ),
        )
        for field, observed, expected in checks
        if not _same_semantic_value(observed, expected)
    )
```

- [ ] **Step 4: Block in both entry points**

In `run_strategy_backtest()`, after the capability check and before factor/data work:

```python
config_issues = validate_backtest_config_matches_spec(config, spec)
if config_issues:
    return StrategyBacktestResult(
        run_id=run_id,
        strategy_id=config.strategy_id,
        strategy_version=spec.version,
        status="BLOCKED",
        reason="CONFIG_SPEC_MISMATCH",
        unsupported_fields=[item.field for item in config_issues],
        capability_issues=[asdict(item) for item in config_issues],
        research_only=True,
        live_trading_allowed=False,
    )
```

In `_run_backtest()`, run the same check immediately after constructing `StrategyBacktestConfig` and before `_backtest_cache_key()`:

```python
config_issues = validate_backtest_config_matches_spec(config, strategy_spec)
if config_issues:
    return _with_backtest_evidence_status({
        "status": "BLOCKED",
        "reason": "CONFIG_SPEC_MISMATCH",
        "unsupported_fields": [item.field for item in config_issues],
        "capability_issues": [asdict(item) for item in config_issues],
        "execution_backend": "factor_rank_baseline_adapter",
        "research_only": True,
        "live_trading_allowed": False,
    })
```

- [ ] **Step 5: Execute from spec-derived locals**

When `spec` exists, derive factor ID, top-N, position cap, cash buffer, rebalance fields, delay, slippage, and single-factor direction from `spec`, not `config`. Use those locals in `SensitivityScenario` and `FactorRankResearchConfig`.

- [ ] **Step 6: Add Agent and direct-adapter block tests**

Assert mismatched `top_n` returns `CONFIG_SPEC_MISMATCH` and monkeypatch data reads to raise `AssertionError` if reached. This proves blocking happens before data access.

- [ ] **Step 7: Verify**

```bash
uv run pytest   tests/unit/strategy/test_backtest_config_spec_consistency.py   tests/unit/agent/test_backtest_config_spec_consistency.py   tests/unit/strategy/test_backtest_config_propagation.py   tests/unit/agent/test_saved_generated_strategy_backtest_guard.py -q
```

- [ ] **Step 8: Commit**

```bash
git add src/qmt_agent_trader/strategy/execution_adapter.py         src/qmt_agent_trader/agent/tools/strategy_tools.py         tests/unit/strategy/test_backtest_config_spec_consistency.py         tests/unit/agent/test_backtest_config_spec_consistency.py
git commit -m "fix(strategy): make spec semantics authoritative"
```

---

## Task 2: Validate Trading-Calendar Completeness

**Files**
- Modify: `src/qmt_agent_trader/data/trading_calendar.py`
- Modify: `tests/unit/data/test_trading_calendar.py`

**Required behavior**
- Every natural date in `[start, end]` has calendar evidence.
- Missing dates raise `TRADING_CALENDAR_PARTIAL_COVERAGE`.
- Invalid date/state values raise `TRADING_CALENDAR_INVALID`.
- Conflicting exchange states for one date raise `TRADING_CALENDAR_CONFLICTING_STATE`.

- [ ] **Step 1: Add failing tests**

```python
def test_partial_calendar_cannot_hide_missing_session(tmp_path):
    lake = data_lake(tmp_path)
    lake.write_parquet(pd.DataFrame([
        {"exchange": "SSE", "cal_date": "20240102", "is_open": 1},
        {"exchange": "SSE", "cal_date": "20240104", "is_open": 1},
    ]), "raw", "tushare/trade_cal")

    with pytest.raises(BacktestDataIntegrityError) as exc:
        load_open_sessions(lake, start="20240102", end="20240104")

    assert exc.value.code == "TRADING_CALENDAR_PARTIAL_COVERAGE"
    assert exc.value.details["missing_dates"] == ["2024-01-03"]


def test_conflicting_calendar_states_raise(tmp_path):
    lake = data_lake(tmp_path)
    lake.write_parquet(pd.DataFrame([
        {"exchange": "SSE", "cal_date": "20240102", "is_open": 1},
        {"exchange": "SZSE", "cal_date": "20240102", "is_open": 0},
    ]), "raw", "tushare/trade_cal")

    with pytest.raises(BacktestDataIntegrityError) as exc:
        load_open_sessions(lake, start="20240102", end="20240102")

    assert exc.value.code == "TRADING_CALENDAR_CONFLICTING_STATE"
```

Also test a non-date `cal_date` and non-binary `is_open`.

- [ ] **Step 2: Implement strict normalization**

```python
from datetime import date, datetime, timedelta


def _parse_boundary(value: str) -> date:
    for fmt in ("%Y%m%d", "%Y-%m-%d"):
        try:
            return datetime.strptime(str(value), fmt).date()
        except ValueError:
            continue
    raise BacktestDataIntegrityError(
        code="TRADING_CALENDAR_INVALID",
        message="calendar boundary is invalid",
        field="trade_cal",
        details={"value": value},
    )


def _natural_dates(start: date, end: date) -> tuple[date, ...]:
    if end < start:
        raise BacktestDataIntegrityError(
            code="TRADING_CALENDAR_INVALID",
            message="calendar end precedes start",
            field="trade_cal",
        )
    return tuple(start + timedelta(days=i) for i in range((end - start).days + 1))
```

Normalize `cal_date` with `errors="coerce"` and `is_open` with `pd.to_numeric`. Reject invalid rows. Compare observed calendar dates with `_natural_dates()`. Do not infer missing dates as closed.

- [ ] **Step 3: Reject conflicts and return open dates**

Group by normalized date, require one unique `is_open` state, then return dates whose state equals `1`.

- [ ] **Step 4: Verify**

```bash
uv run pytest   tests/unit/data/test_trading_calendar.py   tests/unit/backtest/test_expected_sessions.py -q
```

- [ ] **Step 5: Commit**

```bash
git add src/qmt_agent_trader/data/trading_calendar.py         tests/unit/data/test_trading_calendar.py
git commit -m "fix(data): fail on incomplete trading calendars"
```

---

## Task 3: Preserve Ranked Universe Order Before Limits

**Files**
- Modify: `src/qmt_agent_trader/universe/resolver.py`
- Modify: `tests/unit/universe/test_resolver.py`

- [ ] **Step 1: Add failing tests**

```python
def test_ranked_universe_limit_preserves_ranking_order():
    spec = UniverseSpec.model_validate({
        "universe_id": "ranked",
        "name": "Ranked",
        "source": "user_defined",
        "asset_types": ["stock"],
        "selection": {"mode": "all"},
        "ranking": {"field": "avg_amount_20d", "ascending": False},
        "max_symbols": 2,
    })
    frame = pd.DataFrame({
        "symbol": ["000003.SZ", "000001.SZ", "000002.SZ"],
        "avg_amount_20d": [300.0, 200.0, 100.0],
    })

    symbols = _ordered_unique_symbols(frame, spec)
    selected, _ = _apply_limit(symbols, spec=spec, limit=None)

    assert selected == ["000003.SZ", "000001.SZ"]


def test_explicit_symbol_order_is_preserved():
    spec = UniverseSpec.model_validate({
        "universe_id": "explicit",
        "name": "Explicit",
        "source": "user_defined",
        "asset_types": ["stock"],
        "selection": {
            "mode": "explicit_symbols",
            "symbols": ["000003.SZ", "000001.SZ", "000002.SZ"],
        },
    })
    frame = pd.DataFrame({"symbol": spec.selection.symbols})
    assert _ordered_unique_symbols(frame, spec) == spec.selection.symbols
```

- [ ] **Step 2: Implement**

```python
def _ordered_unique_symbols(frame: pd.DataFrame, spec: UniverseSpec) -> list[str]:
    if frame.empty or "symbol" not in frame.columns:
        return []
    ordered = frame
    if spec.ranking is None and spec.selection.mode != "explicit_symbols":
        ordered = frame.sort_values("symbol", kind="stable")
    return list(dict.fromkeys(ordered["symbol"].astype(str).tolist()))
```

Replace the final `sorted(dict.fromkeys(symbols))` in `_resolve_for_date()` with `_ordered_unique_symbols(selected_frame, spec)`.

- [ ] **Step 3: Add a resolver-level test**

Monkeypatch candidate loading with three symbols whose liquidity ranking differs from code order, call `_resolve_for_date()`, then `_apply_limit()`, and assert the ranked top two survive.

- [ ] **Step 4: Verify**

```bash
uv run pytest   tests/unit/universe/test_resolver.py   tests/unit/strategy/test_backtest_rolling_universe.py   tests/unit/strategy/test_backtest_snapshot_universe.py -q
```

- [ ] **Step 5: Commit**

```bash
git add src/qmt_agent_trader/universe/resolver.py         tests/unit/universe/test_resolver.py
git commit -m "fix(universe): preserve ranking before truncation"
```

---

## Task 4: Reject Duplicate Symbol-Date Inputs

**Files**
- Modify: `src/qmt_agent_trader/backtest/research_runner.py`
- Create: `tests/unit/backtest/test_duplicate_inputs.py`

- [ ] **Step 1: Add failing tests**

Test:
1. conflicting duplicate bar rows -> `DUPLICATE_SYMBOL_DATE_BAR`;
2. identical duplicate bar rows -> same error;
3. duplicate factor rows -> `DUPLICATE_FACTOR_SYMBOL_DATE`.

Use a one-day fixture and monkeypatch `compute_factor_frame()` for the factor duplicate case.

- [ ] **Step 2: Implement one validator**

```python
def _require_unique_symbol_trade_dates(
    frame: pd.DataFrame,
    *,
    code: str,
    field: str,
) -> None:
    required = {"symbol", "trade_date"}
    missing = required.difference(frame.columns)
    if missing:
        raise BacktestDataIntegrityError(
            code="INVALID_SYMBOL_DATE_FRAME",
            message="symbol-date frame lacks identity columns",
            field=field,
            details={"missing_columns": sorted(missing)},
        )
    mask = frame.duplicated(["symbol", "trade_date"], keep=False)
    if not mask.any():
        return
    keys = (
        frame.loc[mask, ["symbol", "trade_date"]]
        .drop_duplicates()
        .sort_values(["trade_date", "symbol"])
    )
    raise BacktestDataIntegrityError(
        code=code,
        message="symbol-date identity must be unique",
        symbols=tuple(sorted(keys["symbol"].astype(str).unique())),
        field=field,
        details={
            "duplicate_key_count": len(keys),
            "sample": [
                {
                    "symbol": str(row.symbol),
                    "trade_date": f"{pd.Timestamp(row.trade_date):%Y-%m-%d}",
                }
                for row in keys.head(20).itertuples(index=False)
            ],
        },
    )
```

Call it:
- in `_prepare_bars()` after date normalization;
- immediately after `compute_factor_frame()`.

Remove `match.iloc[0]` duplicate tolerance in `_bar_for_symbol()`. A duplicate reaching that method is a programming invariant failure and should propagate.

- [ ] **Step 3: Verify**

```bash
uv run pytest   tests/unit/backtest/test_duplicate_inputs.py   tests/unit/backtest/test_research_runner_valuation.py   tests/unit/backtest/test_expected_sessions.py -q
```

- [ ] **Step 4: Commit**

```bash
git add src/qmt_agent_trader/backtest/research_runner.py         tests/unit/backtest/test_duplicate_inputs.py
git commit -m "fix(backtest): reject duplicate symbol date inputs"
```

---

## Task 5: Preserve Missing Top-N Overlap as Missing Evidence

**Files**
- Modify: `src/qmt_agent_trader/backtest/research_models.py`
- Modify: `src/qmt_agent_trader/backtest/research_runner.py`
- Modify: `src/qmt_agent_trader/strategy/execution_adapter.py`
- Create: `tests/unit/backtest/test_research_runner_overlap.py`
- Modify: `tests/unit/strategy/test_backtest_diagnostic_wiring.py`

- [ ] **Step 1: Add failing tests**

Assert:
- no comparable selection pairs -> canonical metric is `None`;
- computed overlap is rounded normally;
- `None` overlap yields diagnostic `NOT_COMPUTED`.

- [ ] **Step 2: Change the model and runner**

```python
average_top_n_overlap: float | None = None
```

In the runner:

```python
average_overlap = sum(overlaps) / len(overlaps) if overlaps else None
```

- [ ] **Step 3: Preserve optional evidence**

In `_build_canonical_metrics()`:

```python
"average_top_n_overlap": (
    None
    if result.average_top_n_overlap is None
    else round(result.average_top_n_overlap, 6)
),
```

In `_diagnostic_evidence()`:

```python
churn_report = {}
value = canonical_metrics.get("average_top_n_overlap")
if value is not None:
    churn_report["average_top_n_overlap"] = value
```

Return `churn_report` without the key when unavailable.

- [ ] **Step 4: Verify**

```bash
uv run pytest   tests/unit/backtest/test_research_runner_overlap.py   tests/unit/backtest/test_research_models.py   tests/unit/strategy/test_backtest_diagnostic_wiring.py   tests/unit/strategy/test_backtest_report_schema.py   tests/unit/test_strategy_diagnostics.py -q
```

- [ ] **Step 5: Commit**

```bash
git add src/qmt_agent_trader/backtest/research_models.py         src/qmt_agent_trader/backtest/research_runner.py         src/qmt_agent_trader/strategy/execution_adapter.py         tests/unit/backtest/test_research_runner_overlap.py         tests/unit/strategy/test_backtest_diagnostic_wiring.py
git commit -m "fix(backtest): preserve missing overlap evidence"
```

---

## Task 6: Expose Scheduled Factor-Signal Gaps

**Files**
- Modify: `src/qmt_agent_trader/backtest/research_models.py`
- Modify: `src/qmt_agent_trader/backtest/research_runner.py`
- Create: `tests/unit/backtest/test_research_runner_signal_availability.py`

**Required behavior**
- Each scheduled execution has an available signal or an explicit skipped rebalance point.
- Skip reasons:
  - `factor_signal_date_missing`
  - `factor_signal_all_null`
  - `factor_signal_empty_after_universe_filter`
- No executable scheduled signals -> `NO_EXECUTABLE_FACTOR_SIGNALS`.
- Signal exists but no delayed execution session -> `NO_EXECUTION_SESSION_AFTER_SIGNAL`.

- [ ] **Step 1: Add failing tests**

Test:
1. all factor values NaN -> typed error;
2. initial warm-up gaps followed by valid signals -> completed result with skipped rebalance records;
3. one-day window with delay one -> `NO_EXECUTION_SESSION_AFTER_SIGNAL`;
4. equity point count still equals expected trading-date count.

- [ ] **Step 2: Add completed-run counters**

```python
scheduled_rebalance_count: int = 0
available_signal_count: int = 0
signal_unavailable_count: int = 0
```

- [ ] **Step 3: Prepare signal frames before the loop**

```python
def _prepare_scheduled_signal_frames(
    self,
    signal_dates: tuple[date, ...],
) -> tuple[dict[date, pd.DataFrame], dict[date, str]]:
    raw_by_date = {
        trade_date: frame.copy()
        for trade_date, frame in self.factor_frame.groupby("trade_date")
    }
    available = {}
    unavailable = {}
    for signal_date in signal_dates:
        raw = raw_by_date.get(signal_date)
        if raw is None:
            unavailable[signal_date] = "factor_signal_date_missing"
            continue
        clean = raw.dropna(subset=["factor_value"]).sort_values(
            "factor_value", ascending=False
        )
        if clean.empty:
            unavailable[signal_date] = "factor_signal_all_null"
            continue
        filtered = self._filter_factors_for_universe(clean, signal_date)
        if filtered is None or filtered.empty:
            unavailable[signal_date] = "factor_signal_empty_after_universe_filter"
            continue
        available[signal_date] = filtered
    return available, unavailable
```

- [ ] **Step 4: Fail closed before simulation**

After building the schedule:

```python
if signal_dates and not execution_schedule:
    raise BacktestDataIntegrityError(
        code="NO_EXECUTION_SESSION_AFTER_SIGNAL",
        message="no execution session exists after any signal",
        field="execution_schedule",
        details={"execution_delay_days": scenario.execution_delay_days},
    )
```

Prepare only signal dates that actually map to execution dates. If none are available, raise `NO_EXECUTABLE_FACTOR_SIGNALS` with a date-to-reason mapping.

- [ ] **Step 5: Record skips without duplicating equity points**

For an unavailable signal, append a zero-turnover `ResearchRebalancePoint` with `skipped=True` and the explicit reason. Do not `continue`; fall through to the common end-of-day valuation.

Refactor the existing minimum-turnover skip the same way so every trading date receives exactly one equity point.

- [ ] **Step 6: Emit counters and verify**

```bash
uv run pytest   tests/unit/backtest/test_research_runner_signal_availability.py   tests/unit/backtest/test_research_runner_rebalance.py   tests/unit/backtest/test_research_runner_timeline.py   tests/unit/backtest/test_research_models.py -q
```

- [ ] **Step 7: Commit**

```bash
git add src/qmt_agent_trader/backtest/research_models.py         src/qmt_agent_trader/backtest/research_runner.py         tests/unit/backtest/test_research_runner_signal_availability.py
git commit -m "fix(backtest): expose unavailable factor signals"
```

---

## Task 7: Validate Numeric Inputs and Finish Documentation

**Files**
- Modify: `src/qmt_agent_trader/strategy/execution_adapter.py`
- Modify: `src/qmt_agent_trader/backtest/research_runner.py`
- Modify: `src/qmt_agent_trader/backtest/sensitivity.py`
- Modify: `docs/backtest/factor-rank-adapter.md`
- Create: `tests/unit/backtest/test_research_runner_config_validation.py`
- Modify: `tests/unit/test_sensitivity.py`

- [ ] **Step 1: Constrain Pydantic fields**

```python
initial_cash: float = Field(default=1_000_000, gt=0)
execution_delay_days: int = Field(default=1, ge=1)
slippage_bps: float = Field(default=5.0, ge=0)
top_n: int = Field(default=20, gt=0)
max_single_position_pct: float = Field(default=0.10, gt=0, le=1)
```

- [ ] **Step 2: Add direct dataclass validation**

Add `FactorRankResearchConfig.__post_init__()` validating:
- finite positive initial cash;
- positive top-N;
- position cap in `(0, 1]`;
- cash buffer in `[0, 1)`;
- turnover threshold in `[0, 1]`;
- non-negative rank buffer;
- supported rebalance frequency;
- non-empty, sorted, unique expected dates.

Use exact error messages in tests.

- [ ] **Step 3: Validate sensitivity scenarios**

Add:

```python
def validate_for_factor_rank(self) -> None:
    if self.cost_multiplier <= 0:
        raise ValueError("cost_multiplier must be positive")
    if self.slippage_bps < 0:
        raise ValueError("slippage_bps must be non-negative")
    if self.execution_delay_days < 1:
        raise ValueError("execution_delay_days must be at least one")
    if self.top_n is not None and self.top_n <= 0:
        raise ValueError("top_n must be positive when provided")
    if (
        self.max_single_position_pct is not None
        and not 0 < self.max_single_position_pct <= 1
    ):
        raise ValueError("max_single_position_pct must be in (0, 1]")
```

Call it at the start of `FactorRankResearchRunner.run()`. Update `SensitivityGrid._validate()` so zero execution delay is invalid.

- [ ] **Step 4: Update documentation**

Document:
1. spec authority and `CONFIG_SPEC_MISMATCH`;
2. full natural-date calendar coverage;
3. duplicate symbol-date errors;
4. ranked-order preservation;
5. `None`/`NOT_COMPUTED` overlap;
6. explicit factor-signal skips;
7. no-signal and no-execution-session errors;
8. numeric constraints.

- [ ] **Step 5: Run focused tests**

```bash
uv run pytest   tests/unit/strategy/test_backtest_config_spec_consistency.py   tests/unit/agent/test_backtest_config_spec_consistency.py   tests/unit/data/test_trading_calendar.py   tests/unit/universe/test_resolver.py   tests/unit/backtest/test_duplicate_inputs.py   tests/unit/backtest/test_research_runner_overlap.py   tests/unit/backtest/test_research_runner_signal_availability.py   tests/unit/backtest/test_research_runner_config_validation.py   tests/unit/strategy/test_backtest_diagnostic_wiring.py   tests/unit/strategy/test_backtest_report_schema.py   tests/unit/test_sensitivity.py -q
```

- [ ] **Step 6: Run affected suites**

```bash
uv run pytest   tests/unit/backtest   tests/unit/strategy   tests/unit/universe   tests/unit/data/test_trading_calendar.py   tests/unit/agent/test_backtest_integrity_error_boundary.py   tests/unit/agent/test_saved_generated_strategy_backtest_guard.py   tests/unit/agent/test_backtest_config_spec_consistency.py -q
```

- [ ] **Step 7: Check broad exception handling**

```bash
rg -n "except Exception"   src/qmt_agent_trader/backtest   src/qmt_agent_trader/strategy/execution_adapter.py   src/qmt_agent_trader/agent/tools/strategy_tools.py
```

No broad catch may normalize runner or adapter execution failures.

- [ ] **Step 8: Run repository gate**

```bash
make check
```

- [ ] **Step 9: Commit**

```bash
git add src/qmt_agent_trader/strategy/execution_adapter.py         src/qmt_agent_trader/backtest/research_runner.py         src/qmt_agent_trader/backtest/sensitivity.py         docs/backtest/factor-rank-adapter.md         tests/unit/backtest/test_research_runner_config_validation.py         tests/unit/test_sensitivity.py
git commit -m "fix(backtest): validate final execution invariants"
```

---

# Final Acceptance Checklist

## Strategy semantics
- [ ] StrategySpec is authoritative whenever present.
- [ ] Conflicting factor, portfolio, rebalance, delay, slippage, or direction values return `CONFIG_SPEC_MISMATCH`.
- [ ] Mismatch blocks before cache lookup and data execution.

## Calendar and input integrity
- [ ] Calendar covers every natural date in the requested interval.
- [ ] Missing dates, invalid values, and conflicting states have distinct typed errors.
- [ ] Duplicate bar and factor symbol-date keys are rejected.
- [ ] No arbitrary first duplicate row is selected.

## Universe correctness
- [ ] Ranked order survives deduplication and truncation.
- [ ] Explicit-symbol order is preserved.
- [ ] Unranked non-explicit universes remain deterministic.

## Evidence semantics
- [ ] Missing overlap is `None` and becomes `NOT_COMPUTED`.
- [ ] Every scheduled execution has a signal or explicit skip record.
- [ ] No executable signal and no delayed execution session fail closed.
- [ ] Exactly one equity point exists per expected trading date.

## Numeric and safety constraints
- [ ] Initial cash and all portfolio/scenario values are validated.
- [ ] Integrity errors create no completed report or successful cache entry.
- [ ] Unexpected software exceptions propagate.
- [ ] Research-only and live-trading prohibition remain unchanged.

## Verification
- [ ] Focused tests pass.
- [ ] Affected suites pass.
- [ ] `make check` passes.
- [ ] Documentation matches implementation.

## Explicitly Out of Scope
- Historical extreme-drawdown replay.
- A new full DataLake-to-Agent end-to-end fixture.
- GitHub Actions configuration.

## Expected Merge Decision

Keep `REQUEST_CHANGES` until every checklist item and local verification command passes. Then perform one final review against this plan before merging.
