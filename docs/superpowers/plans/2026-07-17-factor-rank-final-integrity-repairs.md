# Factor-Rank Final Identity, Calendar, Universe, and Provenance Repairs Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove the remaining merge blockers on `codex/factor-rank-backtest-correctness` by making ad-hoc strategy identity explicit end-to-end, making the trading calendar authoritative for universe schedules, requiring complete 20-session liquidity evidence, rejecting malformed point-in-time dates, resolving every requested index independently, and replacing repeated full-dataset hashing with governed content manifests.

**Architecture:** Keep the Agent tool responsible for resolving user intent, but carry the resolved identity mode into the strategy adapter so the adapter cannot reinterpret an ad-hoc request through the Registry. Make `trade_cal` the sole source of official sessions and treat missing market-wide bars on an official open session as a typed data-integrity failure. Keep point-in-time and provenance semantics in focused helpers: strict date parsers, per-index membership maps, and sidecar dataset manifests written by `DataLake`.

**Tech Stack:** Python 3.11+, pandas, Pydantic v2, PyArrow/Parquet, pytest, existing `DataLake`, `StrategyRegistry`, `UniverseResolver`, `AtomicFileStore`, Ruff, mypy, and `uv`.

## Global Constraints

- Target branch: `codex/factor-rank-backtest-correctness`; start from commit `de57c20708da78fbf8d66159450a5e1493f823b8` or a descendant that contains the reviewed changes.
- Save this plan in the repository as `docs/superpowers/plans/2026-07-17-factor-rank-final-integrity-repairs.md`.
- Use TDD for every task: add a failing regression, run it, implement the smallest correct change, rerun, and commit.
- Use one focused commit per task.
- Preserve `research_only=True` and `live_trading_allowed=False` in every result path.
- An ad-hoc factor request must never read a saved strategy, even when a saved strategy uses the same generated ID.
- `trade_cal` is authoritative for official sessions, rolling rebalance dates, and session completeness.
- An official open session with no market-wide bars for the requested asset type is a data-integrity error, not an empty valid universe.
- `avg_amount_20d` and `avg_volume_20d` are valid only with exactly 20 non-null observations over the exact 20-session window.
- Empty optional dates mean an open interval; non-empty malformed dates must fail closed.
- Every requested index code must be resolved independently. Evidence for one index must not hide missing evidence for another.
- Cache correctness remains content-based. The optimization may cache content hashes only through manifests bound to file identity and updated by governed writers.
- Existing datasets without manifests may pay one full hash once; later lookups must use the manifest without rereading the Parquet payload.
- Unexpected programming exceptions propagate. Only typed domain errors are converted at the outer Agent boundary.
- Integrity failures create no completed report and no successful cache entry.
- Do not add or modify GitHub Actions.
- Do not add a new runtime dependency.
- Do not add a historical extreme-drawdown replay.
- Do not add a new full DataLake-to-Agent end-to-end fixture solely for these repairs.

---

## File Responsibility Map

### New files

- `src/qmt_agent_trader/persistence/dataset_manifests.py`
  Defines validated dataset-content manifests and the one-time hash/backfill path used by `DataLake`.

- `tests/unit/strategy/test_backtest_identity_mode.py`
  Proves that the adapter does not query the Registry for `adhoc` or unsaved `inline` execution.

- `tests/unit/persistence/test_dataset_manifests.py`
  Proves manifest correctness, invalidation, and fast repeated fingerprint lookup.

### Existing files to modify

- `src/qmt_agent_trader/agent/tools/strategy_tools.py`
- `src/qmt_agent_trader/strategy/execution_adapter.py`
- `src/qmt_agent_trader/data/trading_calendar.py`
- `src/qmt_agent_trader/universe/resolver.py`
- `src/qmt_agent_trader/universe/pit_metadata.py`
- `src/qmt_agent_trader/data/storage.py`
- `docs/backtest/factor-rank-adapter.md`
- `tests/unit/agent/test_adhoc_factor_strategy_identity.py`
- `tests/unit/data/test_trading_calendar.py`
- `tests/unit/universe/test_exact_session_resolution.py`
- `tests/unit/universe/test_index_membership_asof.py`
- `tests/unit/universe/test_pit_security_master.py`
- `tests/unit/universe/test_resolver.py`
- `tests/unit/agent/test_backtest_cache_provenance.py`

---

### Task 1: Carry Explicit Strategy Identity Mode into the Adapter

**Files:**
- Modify: `src/qmt_agent_trader/agent/tools/strategy_tools.py`
- Modify: `src/qmt_agent_trader/strategy/execution_adapter.py`
- Create: `tests/unit/strategy/test_backtest_identity_mode.py`
- Modify: `tests/unit/agent/test_adhoc_factor_strategy_identity.py`
- Modify: `tests/unit/agent/test_backtest_pre_cache_identity.py`

**Interfaces:**
- Produces: `StrategyIdentityMode = Literal["registry", "inline", "adhoc"]`.
- Produces: `StrategyBacktestConfig.strategy_identity_mode`.
- Produces: `_ResolvedBacktestIntent.strategy_identity_mode`.
- `registry` means the adapter must reload and verify the saved strategy.
- `inline` means the adapter must use the provided spec and must not query the Registry.
- `adhoc` means the adapter must use the temporary spec and must not query the Registry.

- [ ] **Step 1: Strengthen the Agent collision regression**

Replace the saved collision ID in `tests/unit/agent/test_adhoc_factor_strategy_identity.py` with the actual generated ad-hoc ID:

```python
saved_spec = StrategySpec.model_validate(
    {
        "strategy_id": "adhoc_factor_momentum_20d",
        "name": "Saved collision",
        "kind": "FACTOR_RANK_LONG_ONLY",
        "factors": [{"factor_id": "momentum_20d"}],
        "portfolio": {"top_n": 3},
        "rebalance": {"frequency": "monthly"},
    }
)
```

Add the identity-mode assertion:

```python
assert intent.strategy_identity_mode == "adhoc"
```

- [ ] **Step 2: Add adapter-level no-Registry tests**

Create `tests/unit/strategy/test_backtest_identity_mode.py`:

```python
from __future__ import annotations

from pathlib import Path

import pytest

from qmt_agent_trader.data.storage import DataLake
from qmt_agent_trader.strategy import execution_adapter
from qmt_agent_trader.strategy.execution_adapter import (
    StrategyBacktestConfig,
    run_strategy_backtest,
)
from qmt_agent_trader.strategy.models import StrategySpec
from qmt_agent_trader.strategy.registry import StrategyRegistry


def _spec(strategy_id: str) -> StrategySpec:
    return StrategySpec.model_validate(
        {
            "strategy_id": strategy_id,
            "name": strategy_id,
            "kind": "FACTOR_RANK_LONG_ONLY",
            "factors": [{"factor_id": "momentum_20d"}],
            "portfolio": {"top_n": 20},
            "rebalance": {"frequency": "daily"},
        }
    )


@pytest.mark.parametrize("identity_mode", ["adhoc", "inline"])
def test_non_registry_identity_never_queries_registry(
    tmp_path,
    monkeypatch,
    identity_mode: str,
) -> None:
    lake = DataLake(tmp_path / "lake", tmp_path / "research.duckdb")
    registry = StrategyRegistry(tmp_path / "strategies")
    spec = _spec(f"{identity_mode}_strategy")
    config = StrategyBacktestConfig(
        strategy_id=spec.strategy_id,
        strategy_identity_mode=identity_mode,
        strategy_spec=spec,
        factor_name="momentum_20d",
        start_date="20240101",
        end_date="20240131",
    )

    monkeypatch.setattr(
        execution_adapter,
        "_strategy_from_registry",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("Registry must not be queried")
        ),
    )
    monkeypatch.setattr(
        execution_adapter,
        "load_session_window",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("identity stage passed")
        ),
    )

    with pytest.raises(RuntimeError, match="identity stage passed"):
        run_strategy_backtest(
            lake,
            registry,
            config,
            reports_dir=Path(tmp_path / "reports"),
        )
```

Add a Registry-mode test:

```python
def test_registry_identity_requires_saved_strategy(tmp_path) -> None:
    lake = DataLake(tmp_path / "lake", tmp_path / "research.duckdb")
    registry = StrategyRegistry(tmp_path / "strategies")
    spec = _spec("missing_saved_strategy")
    config = StrategyBacktestConfig(
        strategy_id=spec.strategy_id,
        strategy_identity_mode="registry",
        strategy_spec=spec,
        factor_name="momentum_20d",
        start_date="20240101",
        end_date="20240131",
    )

    result = run_strategy_backtest(
        lake,
        registry,
        config,
        reports_dir=Path(tmp_path / "reports"),
    )

    assert result.status == "BLOCKED"
    assert result.reason == "STRATEGY_NOT_FOUND"
```

- [ ] **Step 3: Run the identity regressions and confirm failure**

```bash
uv run pytest \
  tests/unit/agent/test_adhoc_factor_strategy_identity.py \
  tests/unit/strategy/test_backtest_identity_mode.py -q
```

Expected: FAIL because the intent/config do not carry an identity mode and the adapter always queries the Registry.

- [ ] **Step 4: Add the identity mode to the Agent intent**

In `strategy_tools.py`, define:

```python
StrategyIdentityMode = Literal["registry", "inline", "adhoc"]
```

Extend `_ResolvedBacktestIntent`:

```python
@dataclass(frozen=True)
class _ResolvedBacktestIntent:
    strategy_id: str
    strategy_identity_mode: StrategyIdentityMode
    strategy_spec: StrategySpec
    saved_strategy: SavedStrategy | None
    effective_code_path: str | None
    factor_name: str
    requested_factor_ids: tuple[str, ...]
    strategy_frequency: Literal["daily", "weekly", "monthly"]
```

In `_resolve_backtest_intent()`, determine the mode before creating the temporary spec:

```python
if saved_strategy is not None:
    strategy_identity_mode: StrategyIdentityMode = "registry"
elif inline_spec is not None:
    strategy_identity_mode = "inline"
else:
    strategy_identity_mode = "adhoc"
```

Return it in `_ResolvedBacktestIntent`:

```python
return _ResolvedBacktestIntent(
    strategy_id=effective_id,
    strategy_identity_mode=strategy_identity_mode,
    strategy_spec=strategy_spec,
    saved_strategy=saved_strategy,
    effective_code_path=effective_code_path,
    factor_name=factor_name,
    requested_factor_ids=requested_factor_ids,
    strategy_frequency=strategy_frequency,
)
```

- [ ] **Step 5: Add the identity mode to the adapter config**

In `execution_adapter.py`, define the same exported alias:

```python
StrategyIdentityMode = Literal["registry", "inline", "adhoc"]
```

Add the required field to `StrategyBacktestConfig`:

```python
class StrategyBacktestConfig(BaseModel):
    strategy_id: str
    strategy_identity_mode: StrategyIdentityMode
    start_date: str
    end_date: str
    # existing fields remain unchanged
```

Do not give this field a default. Every caller must declare whether Registry identity is intended.

In `_run_backtest()`, pass:

```python
semantic_config = StrategyBacktestConfig(
    strategy_id=strategy_spec.strategy_id,
    strategy_identity_mode=intent.strategy_identity_mode,
    strategy_spec=strategy_spec,
    # existing arguments remain unchanged
)
```

- [ ] **Step 6: Guard Registry lookup in the adapter**

At the start of `run_strategy_backtest()` replace the unconditional lookup with:

```python
saved_strategy = (
    _strategy_from_registry(registry, config.strategy_id)
    if config.strategy_identity_mode == "registry"
    else None
)
if config.strategy_identity_mode == "registry" and saved_strategy is None:
    return StrategyBacktestResult(
        run_id=run_id,
        strategy_id=config.strategy_id,
        strategy_version=(
            config.strategy_spec.version
            if config.strategy_spec is not None
            else "unknown"
        ),
        status="BLOCKED",
        reason="STRATEGY_NOT_FOUND",
        message="registry identity mode requires a saved strategy",
        research_only=True,
        live_trading_allowed=False,
    )
```

Add mode/spec consistency checks:

```python
if config.strategy_identity_mode in {"inline", "adhoc"} and inline_spec is None:
    return StrategyBacktestResult(
        run_id=run_id,
        strategy_id=config.strategy_id,
        strategy_version="unknown",
        status="BLOCKED",
        reason="STRATEGY_IDENTITY_MODE_INVALID",
        message=(
            f"{config.strategy_identity_mode} identity requires an inline StrategySpec"
        ),
        research_only=True,
        live_trading_allowed=False,
    )
```

Keep the existing saved-vs-inline fingerprint check only inside Registry mode:

```python
if (
    config.strategy_identity_mode == "registry"
    and saved_strategy is not None
    and inline_spec is not None
):
    # existing fingerprint comparison
```

- [ ] **Step 7: Migrate direct config callers**

Search every direct constructor:

```bash
rg -n "StrategyBacktestConfig\(" src tests
```

Set exactly one explicit mode in every result:

```python
strategy_identity_mode="adhoc"   # temporary factor baseline
strategy_identity_mode="inline"  # unsaved supplied spec
strategy_identity_mode="registry"  # saved strategy execution
```

Do not add a default to avoid updating tests.

- [ ] **Step 8: Run the identity suite**

```bash
uv run pytest \
  tests/unit/agent/test_adhoc_factor_strategy_identity.py \
  tests/unit/agent/test_backtest_pre_cache_identity.py \
  tests/unit/agent/test_saved_generated_strategy_backtest_guard.py \
  tests/unit/strategy/test_backtest_identity_mode.py \
  tests/unit/strategy/test_saved_strategy_identity.py \
  tests/unit/strategy/test_backtest_config_spec_consistency.py -q
```

Expected: PASS.

- [ ] **Step 9: Commit**

```bash
git add src/qmt_agent_trader/agent/tools/strategy_tools.py \
        src/qmt_agent_trader/strategy/execution_adapter.py \
        tests/unit/agent/test_adhoc_factor_strategy_identity.py \
        tests/unit/agent/test_backtest_pre_cache_identity.py \
        tests/unit/strategy/test_backtest_identity_mode.py
git commit -m "fix(strategy): preserve explicit backtest identity mode"
```

---

### Task 2: Make the Trading Calendar Authoritative for Universe Sessions

**Files:**
- Modify: `src/qmt_agent_trader/data/trading_calendar.py`
- Modify: `src/qmt_agent_trader/universe/resolver.py`
- Modify: `tests/unit/data/test_trading_calendar.py`
- Modify: `tests/unit/universe/test_exact_session_resolution.py`
- Modify: `tests/unit/universe/test_universe_resolver_rolling.py`

**Interfaces:**
- Produces: `open_sessions_between(lake, start, end, exchanges=("SSE", "SZSE")) -> tuple[date, ...]`.
- `latest_open_session_on_or_before()` requires calendar evidence for the requested boundary date.
- `UniverseResolver._rebalance_dates()` derives dates from `trade_cal`, never from bar files.
- Missing all bars for an official open session raises `UNIVERSE_MARKET_SESSION_NOT_READY`.

- [ ] **Step 1: Add boundary-evidence calendar tests**

Append to `tests/unit/data/test_trading_calendar.py`:

```python
def test_latest_open_session_requires_boundary_date_evidence(tmp_path) -> None:
    lake = data_lake(tmp_path)
    lake.write_parquet(
        pd.DataFrame(
            [
                {"exchange": "SSE", "cal_date": "20240102", "is_open": 1},
                {"exchange": "SZSE", "cal_date": "20240102", "is_open": 1},
            ]
        ),
        "raw",
        "tushare/trade_cal",
    )

    with pytest.raises(BacktestDataIntegrityError) as exc_info:
        latest_open_session_on_or_before(lake, as_of="20240103")

    assert exc_info.value.code == "TRADING_CALENDAR_PARTIAL_COVERAGE"
    assert exc_info.value.details["missing_dates"] == ["2024-01-03"]


def test_latest_open_session_allows_closed_boundary_with_evidence(tmp_path) -> None:
    lake = data_lake(tmp_path)
    lake.write_parquet(
        pd.DataFrame(
            [
                {"exchange": "SSE", "cal_date": "20240105", "is_open": 1},
                {"exchange": "SZSE", "cal_date": "20240105", "is_open": 1},
                {"exchange": "SSE", "cal_date": "20240106", "is_open": 0},
                {"exchange": "SZSE", "cal_date": "20240106", "is_open": 0},
            ]
        ),
        "raw",
        "tushare/trade_cal",
    )

    assert latest_open_session_on_or_before(lake, as_of="20240106") == date(2024, 1, 5)
```

- [ ] **Step 2: Replace the fail-open universe test**

In `tests/unit/universe/test_exact_session_resolution.py`, replace the expectation that an official open day with no bars returns `OK + []`:

```python
def test_open_market_session_without_any_bars_fails_closed(tmp_path) -> None:
    lake = DataLake(tmp_path / "lake", tmp_path / "research.duckdb")
    _write_calendar_sessions(lake, ["20240102", "20240103"])
    _write_stock_basic(lake, ["000001.SZ"])
    _write_empty_trade_state_sources(lake)
    lake.write_parquet(
        pd.DataFrame(
            [
                {
                    "ts_code": "000001.SZ",
                    "trade_date": "20240102",
                    "open": 10.0,
                    "high": 10.0,
                    "low": 10.0,
                    "close": 10.0,
                    "vol": 100.0,
                    "amount": 1000.0,
                }
            ]
        ),
        "raw",
        "tushare/daily",
    )

    with pytest.raises(BacktestUniverseIntegrityError) as exc_info:
        UniverseResolver(lake).build(
            _all_stock_spec(),
            mode="snapshot",
            as_of_date="20240103",
        )

    assert exc_info.value.code == "UNIVERSE_MARKET_SESSION_NOT_READY"
    assert exc_info.value.trade_date == "2024-01-03"
```

Add imports for `pytest` and `BacktestUniverseIntegrityError`. Add these helpers near the top of `tests/unit/universe/test_exact_session_resolution.py`:

```python
def _write_calendar_sessions(lake: DataLake, session_keys: list[str]) -> None:
    lake.write_parquet(
        pd.DataFrame(
            [
                {
                    "exchange": exchange,
                    "cal_date": session_key,
                    "is_open": 1,
                }
                for session_key in session_keys
                for exchange in ("SSE", "SZSE")
            ]
        ),
        "raw",
        "tushare/trade_cal",
    )


def _all_stock_spec() -> UniverseSpec:
    return UniverseSpec.model_validate(
        {
            "universe_id": "all_stock",
            "name": "All stock",
            "source": "user_defined",
            "asset_types": ["stock"],
            "selection": {"mode": "all"},
            "filters": {"min_listed_days": 0},
        }
    )
```

- [ ] **Step 3: Add a rolling-calendar source test**

Append to `tests/unit/universe/test_universe_resolver_rolling.py`:

```python
def test_rolling_rebalance_dates_come_from_trade_calendar(tmp_path) -> None:
    lake = DataLake(tmp_path / "lake", tmp_path / "research.duckdb")
    session_keys = ["20240102", "20240103", "20240104", "20240105"]
    lake.write_parquet(
        pd.DataFrame(
            [
                {
                    "exchange": exchange,
                    "cal_date": session_key,
                    "is_open": 1,
                }
                for session_key in session_keys
                for exchange in ("SSE", "SZSE")
            ]
        ),
        "raw",
        "tushare/trade_cal",
    )
    spec = UniverseSpec.model_validate(
        {
            "universe_id": "rolling_stock",
            "name": "Rolling stock",
            "source": "user_defined",
            "asset_types": ["stock"],
            "selection": {"mode": "all"},
            "filters": {"min_listed_days": 0},
        }
    )

    observed = UniverseResolver(lake)._rebalance_dates(
        spec,
        start_date="20240102",
        end_date="20240105",
        frequency="daily",
    )

    assert observed == ["20240102", "20240103", "20240104", "20240105"]
```

Do not write any `daily` rows in this test. It must pass solely from `trade_cal`. Ensure the module imports `pandas as pd`, `DataLake`, `UniverseSpec`, and `UniverseResolver`.

- [ ] **Step 4: Run the new tests and confirm failure**

```bash
uv run pytest \
  tests/unit/data/test_trading_calendar.py \
  tests/unit/universe/test_exact_session_resolution.py \
  tests/unit/universe/test_universe_resolver_rolling.py -q
```

Expected: FAIL because the boundary date is not required, empty exact-session bars are accepted, and rolling dates are sourced from bar files.

- [ ] **Step 5: Add the calendar range helper**

In `trading_calendar.py` add:

```python
def open_sessions_between(
    lake: DataLake,
    *,
    start: str | date,
    end: str | date,
    exchanges: tuple[str, ...] = ("SSE", "SZSE"),
) -> tuple[date, ...]:
    start_date = start if isinstance(start, date) else _parse_boundary(str(start))
    end_date = end if isinstance(end, date) else _parse_boundary(str(end))
    states = _load_normalized_calendar_states(lake, exchanges=exchanges)
    natural_dates = _natural_dates(start_date, end_date)
    missing_dates = [day for day in natural_dates if day not in states]
    if missing_dates:
        raise BacktestDataIntegrityError(
            code="TRADING_CALENDAR_PARTIAL_COVERAGE",
            message="trade calendar lacks evidence for requested universe dates",
            field="trade_cal",
            details={
                "missing_dates": [day.isoformat() for day in missing_dates]
            },
        )
    return tuple(day for day in natural_dates if states[day] == 1)
```

Refactor `load_open_sessions()` to call it directly:

```python
def load_open_sessions(
    lake: DataLake,
    *,
    start: str,
    end: str,
    exchanges: tuple[str, ...] = ("SSE", "SZSE"),
) -> tuple[date, ...]:
    return open_sessions_between(
        lake,
        start=start,
        end=end,
        exchanges=exchanges,
    )
```

- [ ] **Step 6: Require boundary-date evidence**

Update `latest_open_session_on_or_before()`:

```python
boundary = as_of if isinstance(as_of, date) else _parse_boundary(str(as_of))
states = _load_normalized_calendar_states(lake, exchanges=exchanges)
if boundary not in states:
    raise BacktestDataIntegrityError(
        code="TRADING_CALENDAR_PARTIAL_COVERAGE",
        message="trade calendar lacks evidence for requested as-of date",
        field="trade_cal",
        details={"missing_dates": [boundary.isoformat()]},
    )
candidates = [
    day
    for day, is_open in states.items()
    if day <= boundary and is_open == 1
]
```

Keep the existing `TRADING_CALENDAR_EMPTY` behavior when no earlier open session exists.

- [ ] **Step 7: Derive rolling dates from `trade_cal`**

In `universe/resolver.py`, import `open_sessions_between` and replace `_rebalance_dates()`:

```python
def _rebalance_dates(
    self,
    spec: UniverseSpec,
    *,
    start_date: str,
    end_date: str,
    frequency: str,
) -> list[str]:
    sessions = open_sessions_between(
        self.lake,
        start=start_date,
        end=end_date,
    )
    dates = [f"{session:%Y%m%d}" for session in sessions]
    return _period_end_dates(dates, frequency)
```

Delete `_trade_dates()` if it has no remaining callers. Do not retain bar-derived rolling dates as a fallback.

- [ ] **Step 8: Reject an empty official session**

Import `BacktestUniverseIntegrityError` in `universe/resolver.py`. At the end of `_load_recent_bars()`:

```python
bars = load_daily_bars(
    self.lake,
    start=key,
    end=key,
    include_trade_state=True,
    asset_types=list(asset_types),
)
exact = bars[
    bars["trade_date"].eq(effective_date)
    & bars["asset_type"].isin(asset_types)
].reset_index(drop=True)
if exact.empty:
    raise BacktestUniverseIntegrityError(
        code="UNIVERSE_MARKET_SESSION_NOT_READY",
        message="official open session has no market bars for requested assets",
        trade_date=effective_date.isoformat(),
        field="daily_bars",
        details={"asset_types": sorted(set(asset_types))},
    )
return exact
```

- [ ] **Step 9: Run the calendar and universe suites**

```bash
uv run pytest \
  tests/unit/data/test_trading_calendar.py \
  tests/unit/universe/test_exact_session_resolution.py \
  tests/unit/universe/test_universe_resolver_rolling.py \
  tests/unit/universe/test_resolver.py -q
```

Expected: PASS.

- [ ] **Step 10: Commit**

```bash
git add src/qmt_agent_trader/data/trading_calendar.py \
        src/qmt_agent_trader/universe/resolver.py \
        tests/unit/data/test_trading_calendar.py \
        tests/unit/universe/test_exact_session_resolution.py \
        tests/unit/universe/test_universe_resolver_rolling.py
git commit -m "fix(universe): make trade calendar authoritative"
```

---

### Task 3: Require Complete 20-Session Liquidity Evidence

**Files:**
- Modify: `src/qmt_agent_trader/universe/resolver.py`
- Modify: `tests/unit/universe/test_exact_session_resolution.py`
- Modify: `tests/unit/universe/test_resolver.py`

**Interfaces:**
- Produces: `LIQUIDITY_WINDOW_SESSIONS = 20`.
- `_avg_20d_metrics()` returns averages plus observation counts.
- An average is `NaN` unless its corresponding non-null count is exactly 20.
- Universe exclusions distinguish incomplete amount and volume coverage.

- [ ] **Step 1: Add incomplete-window tests**

Append to `tests/unit/universe/test_exact_session_resolution.py`:

```python
def test_nineteen_sessions_do_not_produce_twenty_day_liquidity(tmp_path) -> None:
    lake = DataLake(tmp_path / "lake", tmp_path / "research.duckdb")
    days = [date(2024, 1, 1) + timedelta(days=offset) for offset in range(20)]
    _write_calendar_sessions(lake, [f"{day:%Y%m%d}" for day in days])
    _write_daily_rows(
        lake,
        [
            {
                "ts_code": "000001.SZ",
                "trade_date": f"{day:%Y%m%d}",
                "open": 10.0,
                "high": 10.0,
                "low": 10.0,
                "close": 10.0,
                "vol": 100.0,
                "amount": 1000.0,
            }
            for day in days[1:]
        ],
    )

    observed = UniverseResolver(lake)._avg_20d_metrics(
        f"{days[-1]:%Y%m%d}",
        ["stock"],
    )

    assert observed.loc[0, "amount_observation_count"] == 19
    assert observed.loc[0, "volume_observation_count"] == 19
    assert pd.isna(observed.loc[0, "avg_amount_20d"])
    assert pd.isna(observed.loc[0, "avg_volume_20d"])
```

Add a null-field test:

```python
def test_null_amount_invalidates_only_amount_window(tmp_path) -> None:
    lake = DataLake(tmp_path / "lake", tmp_path / "research.duckdb")
    days = [date(2024, 1, 1) + timedelta(days=offset) for offset in range(20)]
    _write_calendar_sessions(lake, [f"{day:%Y%m%d}" for day in days])
    rows = []
    for index, day in enumerate(days):
        rows.append(
            {
                "ts_code": "000001.SZ",
                "trade_date": f"{day:%Y%m%d}",
                "open": 10.0,
                "high": 10.0,
                "low": 10.0,
                "close": 10.0,
                "vol": 100.0,
                "amount": None if index == 0 else 1000.0,
            }
        )
    _write_daily_rows(lake, rows)

    observed = UniverseResolver(lake)._avg_20d_metrics(
        f"{days[-1]:%Y%m%d}",
        ["stock"],
    )

    assert observed.loc[0, "amount_observation_count"] == 19
    assert observed.loc[0, "volume_observation_count"] == 20
    assert pd.isna(observed.loc[0, "avg_amount_20d"])
    assert observed.loc[0, "avg_volume_20d"] == 100.0
```

Reuse `_write_calendar_sessions()` from Task 2 and add this exact helper to `tests/unit/universe/test_exact_session_resolution.py`:

```python
def _write_daily_rows(
    lake: DataLake,
    rows: list[dict[str, object]],
) -> None:
    lake.write_parquet(
        pd.DataFrame(rows),
        "raw",
        "tushare/daily",
    )
```

- [ ] **Step 2: Add exclusion-reason tests**

Append to `tests/unit/universe/test_resolver.py`:

```python
def test_amount_filter_rejects_incomplete_twenty_day_window() -> None:
    spec = UniverseSpec.model_validate(
        {
            "universe_id": "amount_filter",
            "name": "Amount filter",
            "source": "user_defined",
            "asset_types": ["stock"],
            "selection": {"mode": "all"},
            "filters": {
                "min_listed_days": 0,
                "min_avg_amount_20d": 1.0,
            },
        }
    )
    row = {
        "symbol": "000001.SZ",
        "asset_type": "stock",
        "has_bar_coverage": True,
        "listed_as_of": True,
        "list_date": "20000101",
        "st": False,
        "suspended": False,
        "avg_amount_20d": pd.NA,
        "amount_observation_count": 19,
    }

    reason = UniverseResolver.__new__(UniverseResolver)._exclusion_reason(
        spec,
        row,
        as_of_date="20240131",
    )

    assert reason == "amount_20d_coverage_incomplete"
```

- [ ] **Step 3: Run the regressions and confirm failure**

```bash
uv run pytest \
  tests/unit/universe/test_exact_session_resolution.py \
  tests/unit/universe/test_resolver.py -q
```

Expected: FAIL because averages are currently calculated from any positive number of rows and observation counts are absent.

- [ ] **Step 4: Compute strict observation counts**

In `universe/resolver.py` add:

```python
LIQUIDITY_WINDOW_SESSIONS = 20
```

Replace the aggregation in `_avg_20d_metrics()` with:

```python
bars["amount"] = pd.to_numeric(bars["amount"], errors="coerce")
bars["volume"] = pd.to_numeric(bars["volume"], errors="coerce")
metrics = bars.groupby("symbol", as_index=False).agg(
    observed_session_count=("trade_date", "nunique"),
    avg_amount_20d=("amount", "mean"),
    amount_observation_count=("amount", "count"),
    avg_volume_20d=("volume", "mean"),
    volume_observation_count=("volume", "count"),
)
complete_sessions = metrics["observed_session_count"].eq(
    LIQUIDITY_WINDOW_SESSIONS
)
metrics["avg_amount_20d"] = metrics["avg_amount_20d"].where(
    complete_sessions
    & metrics["amount_observation_count"].eq(LIQUIDITY_WINDOW_SESSIONS)
)
metrics["avg_volume_20d"] = metrics["avg_volume_20d"].where(
    complete_sessions
    & metrics["volume_observation_count"].eq(LIQUIDITY_WINDOW_SESSIONS)
)
return metrics
```

When no frames exist, return all expected columns:

```python
return pd.DataFrame(
    columns=[
        "symbol",
        "observed_session_count",
        "avg_amount_20d",
        "amount_observation_count",
        "avg_volume_20d",
        "volume_observation_count",
    ]
)
```

- [ ] **Step 5: Make incomplete coverage explicit during filtering**

Before the existing average comparisons in `_exclusion_reason()` add:

```python
if filters.min_avg_amount_20d is not None:
    amount_count = _float_or_none(row.get("amount_observation_count"))
    if amount_count != float(LIQUIDITY_WINDOW_SESSIONS):
        return "amount_20d_coverage_incomplete"

if filters.min_avg_volume_20d is not None:
    volume_count = _float_or_none(row.get("volume_observation_count"))
    if volume_count != float(LIQUIDITY_WINDOW_SESSIONS):
        return "volume_20d_coverage_incomplete"
```

Keep the existing `amount_coverage_missing` and `volume_coverage_missing` checks for defensive handling of impossible inconsistent rows.

- [ ] **Step 6: Run the universe tests**

```bash
uv run pytest \
  tests/unit/universe/test_exact_session_resolution.py \
  tests/unit/universe/test_resolver.py \
  tests/unit/universe/test_universe_resolver_snapshot.py \
  tests/unit/universe/test_universe_resolver_rolling.py -q
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add src/qmt_agent_trader/universe/resolver.py \
        tests/unit/universe/test_exact_session_resolution.py \
        tests/unit/universe/test_resolver.py
git commit -m "fix(universe): require complete liquidity windows"
```

---

### Task 4: Reject Malformed Point-in-Time Dates

**Files:**
- Modify: `src/qmt_agent_trader/universe/pit_metadata.py`
- Modify: `tests/unit/universe/test_pit_security_master.py`
- Modify: `tests/unit/universe/test_index_membership_asof.py`

**Interfaces:**
- Produces: `_required_date(values, field, error_code) -> pd.Series`.
- Produces: `_optional_date(values, field, error_code) -> pd.Series`.
- Empty optional values remain open-ended.
- Non-empty malformed values raise the caller-specific typed universe error.

- [ ] **Step 1: Add malformed `delist_date` tests**

Append to `tests/unit/universe/test_pit_security_master.py`:

```python
def test_non_empty_invalid_delist_date_fails_closed() -> None:
    current = pd.DataFrame(
        [
            {
                "ts_code": "000001.SZ",
                "name": "Company",
                "list_date": "20000101",
                "delist_date": "not-a-date",
            }
        ]
    )

    with pytest.raises(BacktestUniverseIntegrityError) as exc_info:
        security_master_asof(current, date(2020, 1, 5))

    assert exc_info.value.code == "UNIVERSE_SECURITY_MASTER_INVALID"
    assert exc_info.value.field == "raw/tushare/stock_basic.delist_date"


def test_empty_delist_date_remains_open_interval() -> None:
    current = pd.DataFrame(
        [
            {
                "ts_code": "000001.SZ",
                "name": "Company",
                "list_date": "20000101",
                "delist_date": None,
            }
        ]
    )

    observed = security_master_asof(current, date(2020, 1, 5))

    assert observed["listed_as_of"].tolist() == [True]
```

Add imports for `pytest` and `BacktestUniverseIntegrityError`.

- [ ] **Step 2: Add malformed index interval tests**

Append to `tests/unit/universe/test_index_membership_asof.py`:

```python
def test_non_empty_invalid_out_date_fails_closed() -> None:
    frame = pd.DataFrame(
        [
            {
                "index_code": "000300.SH",
                "con_code": "000001.SZ",
                "in_date": "20200101",
                "out_date": "bad-date",
            }
        ]
    )

    with pytest.raises(BacktestUniverseIntegrityError) as exc_info:
        index_interval_members_asof(
            frame,
            ["000300.SH"],
            date(2024, 2, 15),
        )

    assert exc_info.value.code == "INDEX_MEMBERSHIP_SOURCE_INVALID"
    assert exc_info.value.field == "raw/tushare/index_member.out_date"


def test_non_empty_invalid_in_date_fails_closed() -> None:
    frame = pd.DataFrame(
        [
            {
                "index_code": "000300.SH",
                "con_code": "000001.SZ",
                "in_date": "bad-date",
                "out_date": None,
            }
        ]
    )

    with pytest.raises(BacktestUniverseIntegrityError) as exc_info:
        index_interval_members_asof(
            frame,
            ["000300.SH"],
            date(2024, 2, 15),
        )

    assert exc_info.value.code == "INDEX_MEMBERSHIP_SOURCE_INVALID"
    assert exc_info.value.field == "raw/tushare/index_member.in_date"
```

- [ ] **Step 3: Run the tests and confirm failure**

```bash
uv run pytest \
  tests/unit/universe/test_pit_security_master.py \
  tests/unit/universe/test_index_membership_asof.py -q
```

Expected: FAIL because optional parsing currently coerces malformed values to `NaT` and required index dates report the wrong error code.

- [ ] **Step 4: Parameterize strict date parsing**

Replace `_required_date()` and `_optional_date()` in `pit_metadata.py` with:

```python
_MISSING_DATE_TOKENS = {"", "nan", "nat", "none", "<na>"}


def _date_text(values: pd.Series) -> tuple[pd.Series, pd.Series]:
    text = values.astype("string").str.strip()
    missing = values.isna() | text.str.lower().isin(_MISSING_DATE_TOKENS)
    return text, missing


def _required_date(
    values: pd.Series,
    *,
    field: str,
    error_code: str,
) -> pd.Series:
    text, missing = _date_text(values)
    parsed = pd.to_datetime(
        text.where(~missing),
        format="mixed",
        errors="coerce",
    )
    invalid = missing | parsed.isna()
    if invalid.any():
        raise BacktestUniverseIntegrityError(
            code=error_code,
            message="point-in-time source contains an invalid required date",
            field=field,
            details={"invalid_row_count": int(invalid.sum())},
        )
    return parsed.dt.date


def _optional_date(
    values: pd.Series,
    *,
    field: str,
    error_code: str,
) -> pd.Series:
    text, missing = _date_text(values)
    parsed = pd.to_datetime(
        text.where(~missing),
        format="mixed",
        errors="coerce",
    )
    invalid = ~missing & parsed.isna()
    if invalid.any():
        raise BacktestUniverseIntegrityError(
            code=error_code,
            message="point-in-time source contains an invalid optional date",
            field=field,
            details={"invalid_row_count": int(invalid.sum())},
        )
    return parsed.dt.date
```

- [ ] **Step 5: Update every call with the correct code**

For `stock_basic`:

```python
data["list_date"] = _required_date(
    data["list_date"],
    field="raw/tushare/stock_basic.list_date",
    error_code="UNIVERSE_SECURITY_MASTER_INVALID",
)
data["delist_date"] = _optional_date(
    data["delist_date"],
    field="raw/tushare/stock_basic.delist_date",
    error_code="UNIVERSE_SECURITY_MASTER_INVALID",
)
```

For `index_weight`:

```python
data["trade_date"] = _required_date(
    data["trade_date"],
    field="raw/tushare/index_weight.trade_date",
    error_code="INDEX_MEMBERSHIP_SOURCE_INVALID",
)
```

For `index_member`:

```python
data["in_date"] = _required_date(
    data["in_date"],
    field="raw/tushare/index_member.in_date",
    error_code="INDEX_MEMBERSHIP_SOURCE_INVALID",
)
data["out_date"] = _optional_date(
    data["out_date"],
    field="raw/tushare/index_member.out_date",
    error_code="INDEX_MEMBERSHIP_SOURCE_INVALID",
)
```

When the optional column is absent, continue creating a `pd.NaT` Series without calling `_optional_date()`.

- [ ] **Step 6: Run PIT metadata tests**

```bash
uv run pytest \
  tests/unit/universe/test_pit_security_master.py \
  tests/unit/universe/test_index_membership_asof.py \
  tests/unit/universe/test_resolver.py -q
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add src/qmt_agent_trader/universe/pit_metadata.py \
        tests/unit/universe/test_pit_security_master.py \
        tests/unit/universe/test_index_membership_asof.py
git commit -m "fix(universe): reject malformed point in time dates"
```

---

### Task 5: Resolve Each Requested Index Independently

**Files:**
- Modify: `src/qmt_agent_trader/universe/pit_metadata.py`
- Modify: `src/qmt_agent_trader/universe/resolver.py`
- Modify: `tests/unit/universe/test_index_membership_asof.py`
- Modify: `tests/unit/universe/test_resolver.py`

**Interfaces:**
- Produces: `index_weight_members_by_code_asof(...) -> dict[str, list[str]]`.
- Produces: `index_interval_members_by_code_asof(...) -> dict[str, list[str]]`.
- Existing flattened helpers remain as compatibility wrappers.
- Resolver precedence is per code: latest `index_weight` snapshot, otherwise active `index_member` intervals.
- A requested code with no evidence in either source raises `INDEX_MEMBERSHIP_NOT_READY`.

- [ ] **Step 1: Add per-code map tests**

Append to `tests/unit/universe/test_index_membership_asof.py`:

```python
def test_index_weight_returns_members_grouped_by_requested_code() -> None:
    frame = pd.DataFrame(
        [
            {
                "index_code": "000300.SH",
                "con_code": "300_A.SZ",
                "trade_date": "20240201",
            },
            {
                "index_code": "000905.SH",
                "con_code": "905_A.SZ",
                "trade_date": "20240202",
            },
        ]
    )

    observed = index_weight_members_by_code_asof(
        frame,
        ["000300.SH", "000905.SH"],
        date(2024, 2, 15),
    )

    assert observed == {
        "000300.SH": ["300_A.SZ"],
        "000905.SH": ["905_A.SZ"],
    }
```

Add the two new functions to the import list.

- [ ] **Step 2: Add mixed-source resolver tests**

Append to `tests/unit/universe/test_resolver.py`:

```python
def test_multiple_indices_resolve_from_independent_sources(tmp_path) -> None:
    lake = DataLake(tmp_path / "lake", tmp_path / "research.duckdb")
    lake.write_parquet(
        pd.DataFrame(
            [
                {
                    "index_code": "000300.SH",
                    "con_code": "000001.SZ",
                    "trade_date": "20240201",
                }
            ]
        ),
        "raw",
        "tushare/index_weight",
    )
    lake.write_parquet(
        pd.DataFrame(
            [
                {
                    "index_code": "000905.SH",
                    "con_code": "000002.SZ",
                    "in_date": "20240101",
                    "out_date": None,
                }
            ]
        ),
        "raw",
        "tushare/index_member",
    )

    observed = UniverseResolver(lake)._index_constituents(
        ["000300.SH", "000905.SH"],
        "20240215",
    )

    assert observed == ["000001.SZ", "000002.SZ"]


def test_missing_one_requested_index_fails_closed(tmp_path) -> None:
    lake = DataLake(tmp_path / "lake", tmp_path / "research.duckdb")
    lake.write_parquet(
        pd.DataFrame(
            [
                {
                    "index_code": "000300.SH",
                    "con_code": "000001.SZ",
                    "trade_date": "20240201",
                }
            ]
        ),
        "raw",
        "tushare/index_weight",
    )

    with pytest.raises(BacktestUniverseIntegrityError) as exc_info:
        UniverseResolver(lake)._index_constituents(
            ["000300.SH", "000905.SH"],
            "20240215",
        )

    assert exc_info.value.code == "INDEX_MEMBERSHIP_NOT_READY"
    assert exc_info.value.details["missing_index_codes"] == ["000905.SH"]
```

Add `BacktestUniverseIntegrityError` to the test module imports. These tests call `_index_constituents()` directly so no market, calendar, or trade-state fixture can mask the source-precedence behavior.

- [ ] **Step 3: Run the tests and confirm failure**

```bash
uv run pytest \
  tests/unit/universe/test_index_membership_asof.py \
  tests/unit/universe/test_resolver.py -q
```

Expected: FAIL because the current resolver returns immediately when any weight source produces members.

- [ ] **Step 4: Return weight membership by code**

In `pit_metadata.py` add:

```python
def index_weight_members_by_code_asof(
    frame: pd.DataFrame,
    index_codes: list[str],
    as_of: date,
) -> dict[str, list[str]]:
    required = {"index_code", "con_code", "trade_date"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise BacktestUniverseIntegrityError(
            code="INDEX_MEMBERSHIP_SOURCE_INVALID",
            message="index_weight lacks required columns",
            field="raw/tushare/index_weight",
            details={"missing_columns": missing},
        )
    data = frame.copy()
    data["index_code"] = data["index_code"].astype(str)
    data["trade_date"] = _required_date(
        data["trade_date"],
        field="raw/tushare/index_weight.trade_date",
        error_code="INDEX_MEMBERSHIP_SOURCE_INVALID",
    )
    requested = set(index_codes)
    data = data[
        data["index_code"].isin(requested)
        & data["trade_date"].map(lambda value: value <= as_of)
    ]
    result: dict[str, list[str]] = {}
    for index_code, group in data.groupby("index_code", sort=True):
        snapshot_date = group["trade_date"].max()
        snapshot = group[group["trade_date"].eq(snapshot_date)]
        require_unique_keys(
            snapshot,
            keys=("index_code", "con_code", "trade_date"),
            code="DUPLICATE_UNIVERSE_SOURCE_KEY",
            field="raw/tushare/index_weight",
        )
        result[str(index_code)] = sorted(
            snapshot["con_code"].astype(str).unique().tolist()
        )
    return result
```

Replace the existing flattened helper body with:

```python
def index_weight_members_asof(
    frame: pd.DataFrame,
    index_codes: list[str],
    as_of: date,
) -> list[str]:
    grouped = index_weight_members_by_code_asof(frame, index_codes, as_of)
    return sorted(
        {
            symbol
            for members in grouped.values()
            for symbol in members
        }
    )
```

- [ ] **Step 5: Return interval membership by code**

Add:

```python
def index_interval_members_by_code_asof(
    frame: pd.DataFrame,
    index_codes: list[str],
    as_of: date,
) -> dict[str, list[str]]:
    required = {"index_code", "con_code", "in_date", "out_date"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise BacktestUniverseIntegrityError(
            code="INDEX_MEMBERSHIP_SOURCE_INVALID",
            message="index_member lacks effective interval columns",
            field="raw/tushare/index_member",
            details={"missing_columns": missing},
        )
    data = frame.copy()
    data["index_code"] = data["index_code"].astype(str)
    data["in_date"] = _required_date(
        data["in_date"],
        field="raw/tushare/index_member.in_date",
        error_code="INDEX_MEMBERSHIP_SOURCE_INVALID",
    )
    data["out_date"] = _optional_date(
        data["out_date"],
        field="raw/tushare/index_member.out_date",
        error_code="INDEX_MEMBERSHIP_SOURCE_INVALID",
    )
    requested = set(index_codes)
    evidence_codes = set(data.loc[data["index_code"].isin(requested), "index_code"])
    active = data[
        data["index_code"].isin(requested)
        & data["in_date"].map(lambda value: value <= as_of)
        & data["out_date"].map(lambda value: pd.isna(value) or value > as_of)
    ]
    require_unique_keys(
        active,
        keys=("index_code", "con_code"),
        code="DUPLICATE_UNIVERSE_SOURCE_KEY",
        field="raw/tushare/index_member",
    )
    return {
        code: sorted(
            active.loc[
                active["index_code"].eq(code),
                "con_code",
            ].astype(str).unique().tolist()
        )
        for code in sorted(evidence_codes)
    }
```

Make `index_interval_members_asof()` flatten this map in the same way as the weight wrapper.

- [ ] **Step 6: Resolve source precedence per code**

In `universe/resolver.py`, import the two map helpers and replace `_index_constituents()` with:

```python
def _index_constituents(
    self,
    index_codes: list[str],
    as_of_date: str,
) -> list[str]:
    as_of = _parse_date(as_of_date)
    normalized_codes = list(dict.fromkeys(str(code) for code in index_codes))
    weight_by_code: dict[str, list[str]] = {}
    member_by_code: dict[str, list[str]] = {}

    weight_path = self.lake.dataset_path("raw", "tushare/index_weight")
    if weight_path.exists():
        weight_by_code = index_weight_members_by_code_asof(
            self.lake.read_parquet("raw", "tushare/index_weight"),
            normalized_codes,
            as_of,
        )

    member_path = self.lake.dataset_path("raw", "tushare/index_member")
    if member_path.exists():
        member_by_code = index_interval_members_by_code_asof(
            self.lake.read_parquet("raw", "tushare/index_member"),
            normalized_codes,
            as_of,
        )

    missing_codes: list[str] = []
    ordered_members: list[str] = []
    for code in normalized_codes:
        if code in weight_by_code:
            members = weight_by_code[code]
        elif code in member_by_code:
            members = member_by_code[code]
        else:
            missing_codes.append(code)
            continue
        ordered_members.extend(members)

    if missing_codes:
        raise BacktestUniverseIntegrityError(
            code="INDEX_MEMBERSHIP_NOT_READY",
            message="one or more requested indices lack as-of membership evidence",
            trade_date=as_of.isoformat(),
            field="index_membership",
            details={"missing_index_codes": missing_codes},
        )

    return [
        normalized
        for item in dict.fromkeys(ordered_members)
        if (normalized := normalize_symbol(item)) is not None
    ]
```

- [ ] **Step 7: Run index and resolver tests**

```bash
uv run pytest \
  tests/unit/universe/test_index_membership_asof.py \
  tests/unit/universe/test_resolver.py \
  tests/unit/universe/test_universe_resolver_snapshot.py \
  tests/unit/universe/test_universe_resolver_rolling.py -q
```

Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add src/qmt_agent_trader/universe/pit_metadata.py \
        src/qmt_agent_trader/universe/resolver.py \
        tests/unit/universe/test_index_membership_asof.py \
        tests/unit/universe/test_resolver.py
git commit -m "fix(universe): resolve every requested index independently"
```

---

### Task 6: Cache Dataset Content Hashes in Governed Manifests

**Files:**
- Create: `src/qmt_agent_trader/persistence/dataset_manifests.py`
- Modify: `src/qmt_agent_trader/data/storage.py`
- Modify: `src/qmt_agent_trader/agent/tools/strategy_tools.py`
- Create: `tests/unit/persistence/test_dataset_manifests.py`
- Modify: `tests/unit/agent/test_backtest_cache_provenance.py`

**Interfaces:**
- Produces: `DatasetContentManifest`.
- Produces: `dataset_manifest_path(dataset_path) -> Path`.
- Produces: `ensure_dataset_content_fingerprint_assume_dataset_locked(path, atomic_store) -> str | None`.
- Produces: `DataLake.dataset_fingerprint(layer, name) -> str | None`.
- Governed writes update the manifest immediately after replacing a Parquet file.
- A manifest is trusted only when size, mtime, ctime, and inode match the current file.
- Missing or stale manifests trigger one full SHA-256 calculation and manifest refresh.

- [ ] **Step 1: Add manifest behavior tests**

Create `tests/unit/persistence/test_dataset_manifests.py`:

```python
from __future__ import annotations

import pandas as pd

from qmt_agent_trader.data.storage import DataLake
from qmt_agent_trader.persistence import dataset_manifests


def test_governed_write_creates_dataset_manifest(tmp_path) -> None:
    lake = DataLake(tmp_path / "lake", tmp_path / "research.duckdb")
    path = lake.write_parquet(
        pd.DataFrame([{"value": 1}]),
        "raw",
        "fixture",
    )

    manifest_path = dataset_manifests.dataset_manifest_path(path)

    assert manifest_path.exists()
    first = lake.dataset_fingerprint("raw", "fixture")
    assert first is not None


def test_second_fingerprint_uses_manifest_without_reading_payload(
    tmp_path,
    monkeypatch,
) -> None:
    lake = DataLake(tmp_path / "lake", tmp_path / "research.duckdb")
    lake.write_parquet(
        pd.DataFrame([{"value": 1}]),
        "raw",
        "fixture",
    )
    first = lake.dataset_fingerprint("raw", "fixture")
    monkeypatch.setattr(
        dataset_manifests,
        "_content_digest",
        lambda _path: (_ for _ in ()).throw(
            AssertionError("payload must not be rehashed")
        ),
    )

    second = lake.dataset_fingerprint("raw", "fixture")

    assert second == first


def test_same_shape_rewrite_changes_manifest_fingerprint(tmp_path) -> None:
    lake = DataLake(tmp_path / "lake", tmp_path / "research.duckdb")
    lake.write_parquet(
        pd.DataFrame([{"value": 1}]),
        "raw",
        "fixture",
    )
    first = lake.dataset_fingerprint("raw", "fixture")

    lake.write_parquet(
        pd.DataFrame([{"value": 2}]),
        "raw",
        "fixture",
    )
    second = lake.dataset_fingerprint("raw", "fixture")

    assert first != second
```

- [ ] **Step 2: Run the tests and confirm failure**

```bash
uv run pytest tests/unit/persistence/test_dataset_manifests.py -q
```

Expected: FAIL because dataset manifests and `DataLake.dataset_fingerprint()` do not exist.

- [ ] **Step 3: Implement the manifest model and helpers**

Create `src/qmt_agent_trader/persistence/dataset_manifests.py`:

```python
"""Content manifests for governed Parquet datasets."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, ValidationError

from qmt_agent_trader.persistence.atomic_files import AtomicFileStore

_CONTENT_CHUNK_BYTES = 1024 * 1024


class DatasetContentManifest(BaseModel):
    schema_version: Literal["1"] = "1"
    dataset_name: str
    size_bytes: int = Field(ge=0)
    mtime_ns: int = Field(ge=0)
    ctime_ns: int = Field(ge=0)
    inode: int = Field(ge=0)
    sha256: str = Field(min_length=64, max_length=64)


def dataset_manifest_path(dataset_path: Path) -> Path:
    return dataset_path.with_suffix(dataset_path.suffix + ".manifest.json")


def ensure_dataset_content_fingerprint_assume_dataset_locked(
    path: Path,
    *,
    atomic_store: AtomicFileStore,
) -> str | None:
    if not path.exists():
        return None
    manifest_path = dataset_manifest_path(path)
    current = _stat_identity(path)
    manifest = _read_manifest(manifest_path)
    if manifest is not None and _matches_current_file(manifest, current):
        return manifest.sha256

    manifest = DatasetContentManifest(
        dataset_name=path.name,
        size_bytes=current["size_bytes"],
        mtime_ns=current["mtime_ns"],
        ctime_ns=current["ctime_ns"],
        inode=current["inode"],
        sha256=_content_digest(path),
    )
    payload = manifest.model_dump(mode="json")
    atomic_store.write_json(manifest_path, payload)
    return manifest.sha256


def _read_manifest(path: Path) -> DatasetContentManifest | None:
    if not path.exists():
        return None
    try:
        return DatasetContentManifest.model_validate(
            json.loads(path.read_text(encoding="utf-8"))
        )
    except (OSError, ValueError, ValidationError):
        return None


def _stat_identity(path: Path) -> dict[str, int]:
    stat = path.stat()
    return {
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "ctime_ns": int(stat.st_ctime_ns),
        "inode": int(getattr(stat, "st_ino", 0)),
    }


def _matches_current_file(
    manifest: DatasetContentManifest,
    current: dict[str, int],
) -> bool:
    return (
        manifest.size_bytes == current["size_bytes"]
        and manifest.mtime_ns == current["mtime_ns"]
        and manifest.ctime_ns == current["ctime_ns"]
        and manifest.inode == current["inode"]
    )


def _content_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(_CONTENT_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()
```

Do not duplicate the content-hash implementation in `provenance.py` after this task; Task 6 Step 7 consolidates it.

- [ ] **Step 4: Add a governed `DataLake` fingerprint method**

In `data/storage.py` import:

```python
from qmt_agent_trader.persistence.dataset_manifests import (
    ensure_dataset_content_fingerprint_assume_dataset_locked,
)
```

Add:

```python
def dataset_fingerprint(self, layer: str, name: str) -> str | None:
    path = self.dataset_path(layer, name)
    if not path.exists():
        return None
    with self.lock_manager.resource_lock(path):
        return ensure_dataset_content_fingerprint_assume_dataset_locked(
            path,
            atomic_store=self.atomic_store,
        )
```

- [ ] **Step 5: Update manifests after governed writes**

Change `write_parquet()` so data replacement and manifest refresh occur under the dataset lock:

```python
def write_parquet(self, frame: pd.DataFrame, layer: str, name: str) -> Path:
    path = self.dataset_path(layer, name)
    path.parent.mkdir(parents=True, exist_ok=True)
    writable = frame
    if len(frame.columns) == 0:
        writable = pd.DataFrame({"_empty": pd.Series(dtype="bool")})
    with self.lock_manager.resource_lock(path):
        self.atomic_store.write_parquet_assume_locked(path, writable)
        ensure_dataset_content_fingerprint_assume_dataset_locked(
            path,
            atomic_store=self.atomic_store,
        )
    return path
```

In `write_incremental_parquet()`, immediately after `write_parquet_assume_locked()` add:

```python
ensure_dataset_content_fingerprint_assume_dataset_locked(
    path,
    atomic_store=self.atomic_store,
)
```

The manifest itself uses its own manifest-path lock through `AtomicFileStore.write_json()`. Do not call `write_json_assume_locked()` because the held lock is for the Parquet path, not the sidecar path.

- [ ] **Step 6: Run manifest tests**

```bash
uv run pytest \
  tests/unit/persistence/test_dataset_manifests.py \
  tests/unit/persistence/test_provenance_content_hash.py \
  tests/unit/persistence/test_infrastructure.py -q
```

Expected: PASS.

- [ ] **Step 7: Use dataset manifests in backtest provenance**

In `strategy_tools.py` bump versions:

```python
BACKTEST_CACHE_SCHEMA_VERSION = "factor-rank-v4"
BACKTEST_ENGINE_SEMANTIC_VERSION = "2026-07-universe-integrity-v3"
```

Replace dataset fingerprint construction:

```python
dataset_fingerprints = {
    name: lake.dataset_fingerprint("raw", name)
    for name in sorted(dataset_names)
}
```

Keep `fingerprint_path_tree()` for strategy code and factor implementation files, because those are small non-dataset artifacts.

- [ ] **Step 8: Add a provenance fast-path regression**

Append to `tests/unit/agent/test_backtest_cache_provenance.py`:

```python
def test_backtest_provenance_reuses_dataset_manifests(
    tmp_path,
    monkeypatch,
) -> None:
    lake = DataLake(tmp_path / "lake", tmp_path / "research.duckdb")
    for dataset in (
        "tushare/daily",
        "tushare/fund_daily",
        "tushare/trade_cal",
        "tushare/suspend_d",
        "tushare/stk_limit",
        "tushare/namechange",
        "tushare/stock_basic",
        "tushare/index_weight",
        "tushare/index_member",
    ):
        lake.write_parquet(
            pd.DataFrame([{"dataset": dataset}]),
            "raw",
            dataset,
        )
    config = _fixture_config()
    strategy_tools._backtest_provenance_manifest(
        lake,
        config=config,
        requested_factor_ids=[],
        saved_strategy=None,
        effective_code_path=None,
        resolved_universe={"symbols": ["000001.SZ"]},
    )
    monkeypatch.setattr(
        dataset_manifests,
        "_content_digest",
        lambda _path: (_ for _ in ()).throw(
            AssertionError("dataset payload must not be rehashed")
        ),
    )

    strategy_tools._backtest_provenance_manifest(
        lake,
        config=config,
        requested_factor_ids=[],
        saved_strategy=None,
        effective_code_path=None,
        resolved_universe={"symbols": ["000001.SZ"]},
    )
```

Import `pandas as pd` and `dataset_manifests`. Define `_fixture_config()` as a complete `StrategyBacktestConfig` with `strategy_identity_mode="adhoc"`, an ad-hoc `StrategySpec`, and the existing required fields.

- [ ] **Step 9: Run cache tests**

```bash
uv run pytest \
  tests/unit/persistence/test_dataset_manifests.py \
  tests/unit/persistence/test_provenance_content_hash.py \
  tests/unit/agent/test_backtest_cache_provenance.py -q
```

Expected: PASS.

- [ ] **Step 10: Commit**

```bash
git add src/qmt_agent_trader/persistence/dataset_manifests.py \
        src/qmt_agent_trader/data/storage.py \
        src/qmt_agent_trader/agent/tools/strategy_tools.py \
        tests/unit/persistence/test_dataset_manifests.py \
        tests/unit/agent/test_backtest_cache_provenance.py
git commit -m "perf(cache): persist governed dataset content hashes"
```

---

### Task 7: Update Contracts and Run Final Verification

**Files:**
- Modify: `docs/backtest/factor-rank-adapter.md`

**Interfaces:**
- Documentation must match the final identity, calendar, liquidity, index, and provenance contracts.
- No GitHub Actions changes are permitted.

- [ ] **Step 1: Document explicit identity modes**

Add:

```markdown
## Strategy identity modes

Every backtest config declares one identity mode:

- `registry`: reload and verify the saved Registry strategy;
- `inline`: execute the supplied unsaved StrategySpec without Registry lookup;
- `adhoc`: execute a temporary factor baseline without Registry lookup.

Generated ad-hoc IDs are cache/report identifiers only. They can never cause a
saved strategy with the same text ID to be loaded.
```

- [ ] **Step 2: Document calendar and exact-session behavior**

Add:

```markdown
## Universe session authority

`trade_cal` is authoritative for snapshot effective sessions and rolling
rebalance dates. The requested as-of date must itself have calendar evidence,
including closed weekend and holiday records. An official open session with no
market-wide bars for the requested asset type raises
`UNIVERSE_MARKET_SESSION_NOT_READY`; it is not reported as a valid empty
universe.
```

- [ ] **Step 3: Document strict 20-session metrics**

Add:

```markdown
## Liquidity-window completeness

`avg_amount_20d` and `avg_volume_20d` require exactly 20 official sessions and
20 non-null observations for the corresponding field. Short listings, missing
sessions, suspension gaps, and null values leave the metric unavailable and
produce explicit observation-count evidence.
```

- [ ] **Step 4: Document PIT date and multi-index behavior**

Add:

```markdown
## Point-in-time date and index evidence

Empty `delist_date` and `out_date` values represent open intervals. Non-empty
malformed dates are invalid source evidence and fail closed.

Each requested index code is resolved independently. The resolver uses the
latest `index_weight` snapshot for that code when available, otherwise active
`index_member` intervals. Missing evidence for any requested code raises
`INDEX_MEMBERSHIP_NOT_READY`.
```

- [ ] **Step 5: Document cache manifests**

Replace the cache version text with:

```markdown
Cache schema `factor-rank-v4` uses content SHA-256 values. Governed DataLake
writes persist a sidecar content manifest bound to file size, mtime, ctime, and
inode. A missing or stale manifest triggers one full rehash and refresh;
subsequent cache-key construction reads the small manifest instead of the full
Parquet payload.
```

- [ ] **Step 6: Run all focused tests**

```bash
uv run pytest \
  tests/unit/agent/test_adhoc_factor_strategy_identity.py \
  tests/unit/agent/test_backtest_pre_cache_identity.py \
  tests/unit/agent/test_backtest_cache_provenance.py \
  tests/unit/strategy/test_backtest_identity_mode.py \
  tests/unit/data/test_trading_calendar.py \
  tests/unit/universe/test_exact_session_resolution.py \
  tests/unit/universe/test_index_membership_asof.py \
  tests/unit/universe/test_pit_security_master.py \
  tests/unit/universe/test_resolver.py \
  tests/unit/persistence/test_dataset_manifests.py -q
```

Expected: PASS.

- [ ] **Step 7: Run affected subsystem suites**

```bash
uv run pytest \
  tests/unit/backtest \
  tests/unit/data \
  tests/unit/factors \
  tests/unit/strategy \
  tests/unit/universe \
  tests/unit/persistence \
  tests/unit/agent/test_adhoc_factor_strategy_identity.py \
  tests/unit/agent/test_backtest_pre_cache_identity.py \
  tests/unit/agent/test_backtest_cache_provenance.py \
  tests/integration/test_factor_rank_backtest_correctness.py -q
```

Expected: PASS.

- [ ] **Step 8: Verify Registry access is mode-gated**

```bash
rg -n "_strategy_from_registry|strategy_identity_mode" \
  src/qmt_agent_trader/strategy/execution_adapter.py \
  src/qmt_agent_trader/agent/tools/strategy_tools.py
```

Confirm:

1. Agent intent always sets an explicit mode.
2. Adapter Registry lookup appears only under `strategy_identity_mode == "registry"`.
3. No generated ad-hoc ID is used for Registry lookup.

- [ ] **Step 9: Verify calendar-derived universe dates**

```bash
rg -n "_trade_dates|open_sessions_between|latest_open_session_on_or_before" \
  src/qmt_agent_trader/universe \
  src/qmt_agent_trader/data/trading_calendar.py
```

Confirm rolling rebalance dates do not read `daily` or `fund_daily` date lists.

- [ ] **Step 10: Verify strict liquidity windows**

```bash
rg -n "avg_amount_20d|avg_volume_20d|observation_count|LIQUIDITY_WINDOW_SESSIONS" \
  src/qmt_agent_trader/universe/resolver.py
```

Confirm averages are masked unless the corresponding count equals 20.

- [ ] **Step 11: Verify malformed optional dates cannot fail open**

```bash
rg -n "errors=\"coerce\"|_optional_date|_required_date" \
  src/qmt_agent_trader/universe/pit_metadata.py
```

Confirm every `errors="coerce"` result is followed by explicit invalid non-empty detection.

- [ ] **Step 12: Verify multi-index resolution**

```bash
rg -n "index_weight_members_by_code_asof|index_interval_members_by_code_asof|INDEX_MEMBERSHIP_NOT_READY" \
  src/qmt_agent_trader/universe
```

Confirm the resolver loops over every requested index code and does not return early after the first source match.

- [ ] **Step 13: Verify dataset fingerprints avoid repeated full reads**

```bash
rg -n "dataset_fingerprint|ensure_dataset_content_fingerprint|fingerprint_path_tree" \
  src/qmt_agent_trader/data/storage.py \
  src/qmt_agent_trader/agent/tools/strategy_tools.py \
  src/qmt_agent_trader/persistence
```

Confirm:

1. raw dataset provenance uses `lake.dataset_fingerprint()`;
2. strategy/factor code still uses `fingerprint_path_tree()`;
3. a valid manifest returns its stored SHA without calling `_content_digest()`.

- [ ] **Step 14: Verify no CI changes entered the branch**

```bash
git diff --name-only main...HEAD -- .github/workflows
```

Expected: no output.

- [ ] **Step 15: Run repository gates**

```bash
make check
```

Expected: exit code `0`.

- [ ] **Step 16: Commit documentation**

```bash
git add docs/backtest/factor-rank-adapter.md
git commit -m "docs(backtest): document final integrity contracts"
```

---

## Final Acceptance Checklist

### Strategy identity

- [ ] Every `StrategyBacktestConfig` has an explicit identity mode.
- [ ] `adhoc` never queries `StrategyRegistry`.
- [ ] Unsaved `inline` never queries `StrategyRegistry`.
- [ ] `registry` fails with `STRATEGY_NOT_FOUND` when the saved strategy is absent.
- [ ] Saved spec fingerprint and generated-code checks remain mandatory in Registry mode.
- [ ] A saved strategy named `adhoc_factor_<factor>` cannot affect a pure factor request.

### Calendar and session integrity

- [ ] The requested as-of date must have a `trade_cal` row.
- [ ] A closed boundary date with evidence can resolve to the previous open session.
- [ ] Rolling rebalance dates come only from `trade_cal`.
- [ ] An official open session with no requested-asset bars raises `UNIVERSE_MARKET_SESSION_NOT_READY`.
- [ ] Missing market-wide data cannot become `OK + empty symbols`.

### Liquidity evidence

- [ ] The raw read is bounded to the exact 20 official sessions.
- [ ] `avg_amount_20d` requires 20 non-null amount observations.
- [ ] `avg_volume_20d` requires 20 non-null volume observations.
- [ ] Observation counts are returned and merged into candidate evidence.
- [ ] Incomplete windows have explicit exclusion reasons.

### PIT dates and index membership

- [ ] Empty optional dates remain open intervals.
- [ ] Non-empty malformed `delist_date` fails closed.
- [ ] Non-empty malformed `in_date` and `out_date` fail closed.
- [ ] Error codes distinguish security-master and index-membership sources.
- [ ] Every requested index code is resolved independently.
- [ ] Per-code source precedence is weight snapshot, then interval membership.
- [ ] Missing evidence for one index cannot be hidden by evidence for another.

### Cache provenance

- [ ] Cache schema is `factor-rank-v4`.
- [ ] Content SHA-256 remains the provenance value.
- [ ] Governed writes refresh dataset manifests.
- [ ] Missing or stale manifests trigger one full hash.
- [ ] A valid manifest avoids rereading the Parquet payload.
- [ ] Same-shape governed rewrites change the fingerprint.
- [ ] Code and Registry artifacts retain direct content hashing.

### Safety and verification

- [ ] Integrity errors create no completed report.
- [ ] Integrity errors create no successful cache entry.
- [ ] Unexpected software exceptions propagate.
- [ ] `research_only=True`.
- [ ] `live_trading_allowed=False`.
- [ ] Focused tests pass.
- [ ] Affected subsystem suites pass.
- [ ] Existing factor-rank integration test passes.
- [ ] `make check` passes.
- [ ] No GitHub Actions files changed.
- [ ] Documentation matches the implementation.

## Expected Merge Decision

Keep `REQUEST_CHANGES` until every acceptance item and local verification command passes. After implementation, perform one final static review of the current branch head and inspect the real `make check` output before merging.
