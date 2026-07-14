# Factor-Rank Backtest Final Blockers Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove the remaining correctness blockers on `codex/factor-rank-backtest-correctness` so execution-state evidence is time-correct and asset-aware, saved strategy identity cannot be bypassed through cache, warm-up history is validated and excluded from diagnostics, ambiguous source rows fail closed, and the runner never manufactures missing inputs.

**Architecture:** Establish one strict canonical execution-input contract shared by the data loader, universe resolver, adapter, and runner. Resolve strategy identity and temporary strategy semantics before universe work or cache access. Build cache keys from a provenance manifest that fingerprints every effective input. Treat warm-up history as a separate factor-input window with explicit coverage validation and performance-only diagnostic views.

**Tech Stack:** Python 3.11+, pandas, Pydantic v2, dataclasses, pytest, existing `DataLake`, `FactorRegistry`, `StrategyRegistry`, `UniverseResolver`, factor-rank runner, Ruff, mypy, and `uv`.

## Global Constraints

- Target branch: `codex/factor-rank-backtest-correctness`; continue from its current head.
- Save this plan as `docs/superpowers/plans/2026-07-14-factor-rank-backtest-final-blockers.md`.
- Use one focused commit per task.
- Follow TDD: failing regression, verify failure, minimal implementation, verify pass, commit.
- Opening execution decisions may use only information available at the opening auction.
- Unknown execution state must never be converted to `False`.
- Stock and ETF execution-state evidence must be handled by separate asset-aware paths.
- Until an ETF limit-state source is implemented, ETF backtests must return a typed unsupported-state error rather than reuse stock-only `stk_limit`.
- Registry identity and generated-code guards must run before universe resolution, cache-key construction, and cache lookup.
- A cache key must fingerprint all effective data, strategy, universe, factor, and engine-semantic inputs.
- Warm-up bars may be used for factor calculation only; they must not enter trades, equity, metrics, IC, walk-forward, or other reported performance evidence.
- Ambiguous duplicate source rows must raise typed errors; never select first/last by storage order.
- Only the outer Agent-tool boundary converts typed `BacktestIntegrityError` subclasses to structured `ERROR`.
- Unexpected programming exceptions propagate.
- Integrity failures create no completed report and no successful cache entry.
- Preserve `research_only=True` and `live_trading_allowed=False`.
- Do not add a runtime dependency.
- Do not add or modify GitHub Actions.
- Do not reproduce the historical extreme-drawdown run.
- Do not create a new full DataLake-to-Agent integration suite. Small production-boundary component tests are required.

---

## File Responsibility Map

### New files

- `src/qmt_agent_trader/data/trade_state.py`: strict stock opening-state normalization and asset-aware errors.
- `src/qmt_agent_trader/persistence/provenance.py`: stable directory/file fingerprints for cache provenance.
- `tests/unit/data/test_opening_trade_state.py`
- `tests/unit/data/test_asset_aware_trade_state.py`
- `tests/unit/agent/test_backtest_pre_cache_identity.py`
- `tests/unit/agent/test_backtest_cache_provenance.py`
- `tests/unit/backtest/test_warmup_coverage.py`
- `tests/unit/strategy/test_performance_diagnostic_window.py`
- `tests/unit/factors/test_asof_ambiguity.py`
- `tests/unit/backtest/test_runner_input_contract.py`

### Existing files to modify

- `src/qmt_agent_trader/data/bars.py`
- `src/qmt_agent_trader/data/trading_calendar.py`
- `src/qmt_agent_trader/factors/input_panel.py`
- `src/qmt_agent_trader/agent/tools/strategy_tools.py`
- `src/qmt_agent_trader/strategy/execution_adapter.py`
- `src/qmt_agent_trader/backtest/research_runner.py`
- `src/qmt_agent_trader/backtest/research_models.py`
- `src/qmt_agent_trader/universe/resolver.py`
- `docs/backtest/factor-rank-adapter.md`

---

## Task 1: Define a Strict Opening-Execution State Contract

**Files:**
- Create: `src/qmt_agent_trader/data/trade_state.py`
- Modify: `src/qmt_agent_trader/data/bars.py`
- Modify: `src/qmt_agent_trader/strategy/execution_adapter.py`
- Modify: `src/qmt_agent_trader/backtest/research_runner.py`
- Create: `tests/unit/data/test_opening_trade_state.py`

**Interfaces:**
- Produce `OPENING_TRADE_STATE_COLUMNS`.
- Produce `normalize_stock_opening_trade_state(...) -> pd.DataFrame`.
- Canonical execution columns: `suspended`, `st`, `limit_up_at_open`, `limit_down_at_open`.
- Closing prices must not affect opening execution eligibility.

- [ ] **Step 1: Write failing opening-limit tests**

```python
from datetime import date
import pandas as pd
import pytest

from qmt_agent_trader.backtest.errors import BacktestDataIntegrityError
from qmt_agent_trader.data.trade_state import normalize_stock_opening_trade_state


def bars(open_price: float, close_price: float) -> pd.DataFrame:
    return pd.DataFrame([{
        "symbol": "000001.SZ",
        "trade_date": date(2024, 1, 2),
        "open": open_price,
        "close": close_price,
    }])


def limits() -> pd.DataFrame:
    return pd.DataFrame([{
        "ts_code": "000001.SZ",
        "trade_date": "20240102",
        "up_limit": 11.0,
        "down_limit": 9.0,
    }])


def test_close_limit_does_not_block_opening_buy() -> None:
    result = normalize_stock_opening_trade_state(
        bars(10.0, 11.0),
        suspend=pd.DataFrame(columns=["ts_code", "trade_date", "suspend_type"]),
        stk_limit=limits(),
        namechange=pd.DataFrame(columns=["ts_code", "name", "start_date", "end_date"]),
    )
    assert result.loc[0, "limit_up_at_open"] is False


def test_open_at_upper_limit_blocks_opening_buy() -> None:
    result = normalize_stock_opening_trade_state(
        bars(11.0, 10.5),
        suspend=pd.DataFrame(columns=["ts_code", "trade_date", "suspend_type"]),
        stk_limit=limits(),
        namechange=pd.DataFrame(columns=["ts_code", "name", "start_date", "end_date"]),
    )
    assert result.loc[0, "limit_up_at_open"] is True


@pytest.mark.parametrize("column,value", [
    ("up_limit", None),
    ("up_limit", 0.0),
    ("up_limit", float("inf")),
    ("down_limit", None),
    ("down_limit", -1.0),
])
def test_invalid_limit_price_fails_closed(column, value) -> None:
    source = limits()
    source.loc[0, column] = value
    with pytest.raises(BacktestDataIntegrityError) as exc:
        normalize_stock_opening_trade_state(
            bars(10.0, 10.0),
            suspend=pd.DataFrame(columns=["ts_code", "trade_date", "suspend_type"]),
            stk_limit=source,
            namechange=pd.DataFrame(columns=["ts_code", "name", "start_date", "end_date"]),
        )
    assert exc.value.code == "INVALID_TRADE_STATE_SOURCE"
```

- [ ] **Step 2: Verify failure**

```bash
uv run pytest tests/unit/data/test_opening_trade_state.py -q
```

Expected: import failure for `qmt_agent_trader.data.trade_state`.

- [ ] **Step 3: Implement strict source validation and opening flags**

Create `trade_state.py` with these public members:

```python
OPENING_TRADE_STATE_COLUMNS = (
    "suspended",
    "st",
    "limit_up_at_open",
    "limit_down_at_open",
)


def normalize_stock_opening_trade_state(
    bars: pd.DataFrame,
    *,
    suspend: pd.DataFrame,
    stk_limit: pd.DataFrame,
    namechange: pd.DataFrame,
) -> pd.DataFrame:
    limits = _normalize_stock_limits(stk_limit)
    suspend_events = _normalize_suspend_events(suspend)
    st_periods = _normalize_namechange_periods(namechange)
    result = bars.copy()
    result["trade_date"] = _coerce_dates(result["trade_date"], field="bars.trade_date")
    require_unique_symbol_dates(
        result,
        symbol_column="symbol",
        date_column="trade_date",
        code="DUPLICATE_SYMBOL_DATE_BAR",
        field="bars",
    )
    result = result.merge(limits, on=["symbol", "trade_date"], how="left", validate="one_to_one")
    missing = result["up_limit"].isna() | result["down_limit"].isna()
    if missing.any():
        raise BacktestDataIntegrityError(
            code="TRADE_STATE_PARTIAL_COVERAGE",
            message="stock limit source does not cover every stock bar",
            field="raw/tushare/stk_limit",
            symbols=tuple(sorted(result.loc[missing, "symbol"].astype(str).unique())),
        )
    tolerance = 1e-6
    result["limit_up_at_open"] = pd.to_numeric(result["open"], errors="coerce") >= result["up_limit"] - tolerance
    result["limit_down_at_open"] = pd.to_numeric(result["open"], errors="coerce") <= result["down_limit"] + tolerance
    suspended_keys = set(zip(suspend_events["symbol"], suspend_events["trade_date"], strict=False))
    result["suspended"] = [
        (str(symbol), day) in suspended_keys
        for symbol, day in zip(result["symbol"].astype(str), result["trade_date"], strict=False)
    ]
    result["st"] = _historical_st_mask(result, st_periods)
    for column in OPENING_TRADE_STATE_COLUMNS:
        if result[column].isna().any():
            raise BacktestDataIntegrityError(
                code="UNKNOWN_EXECUTION_STATE",
                message="opening execution state contains unknown values",
                field=column,
            )
        result[column] = result[column].astype(bool)
    result.attrs["trade_state_quality"] = {
        "asset_type": "stock",
        "execution_time": "open",
        "suspended": {"source": "raw/tushare/suspend_d", "complete": True},
        "st": {"source": "raw/tushare/namechange", "complete": True},
        "limit_up_at_open": {"source": "raw/tushare/stk_limit", "complete": True},
        "limit_down_at_open": {"source": "raw/tushare/stk_limit", "complete": True},
    }
    return result.drop(columns=["up_limit", "down_limit"])
```

Private validators must enforce:

- `suspend_d`: `ts_code`, `trade_date`; unique symbol-date; valid dates.
- `stk_limit`: `ts_code`, `trade_date`, `up_limit`, `down_limit`; unique symbol-date; finite positive prices; `down_limit < up_limit`.
- `namechange`: `ts_code`, `name`, `start_date`, `end_date`; valid intervals; unique period records.

Each invalid source raises `BacktestDataIntegrityError(code="INVALID_TRADE_STATE_SOURCE", ...)` or `DUPLICATE_TRADE_STATE_INPUT`.

- [ ] **Step 4: Route loaders and runner to opening fields**

In `bars.py`, replace the old close-aware enrichment with `normalize_stock_opening_trade_state()`.

In `execution_adapter.py`, replace base fields `limit_up` and `limit_down` with `limit_up_at_open` and `limit_down_at_open`.

In `research_runner.py`:

```python
@staticmethod
def _can_execute(bar: pd.Series, side: Side) -> bool:
    if bool(bar["suspended"]):
        return False
    if side == Side.BUY:
        return not bool(bar["st"]) and not bool(bar["limit_up_at_open"])
    return not bool(bar["limit_down_at_open"])
```

Do not read closing limit flags for opening execution.

- [ ] **Step 5: Verify and commit**

```bash
uv run pytest \
  tests/unit/data/test_opening_trade_state.py \
  tests/unit/data/test_trade_state_evidence.py \
  tests/unit/backtest/test_research_runner_valuation.py -q

git add src/qmt_agent_trader/data/trade_state.py \
        src/qmt_agent_trader/data/bars.py \
        src/qmt_agent_trader/strategy/execution_adapter.py \
        src/qmt_agent_trader/backtest/research_runner.py \
        tests/unit/data/test_opening_trade_state.py

git commit -m "fix(backtest): use opening-only trade state"
```

---

## Task 2: Make Trade-State Loading Asset-Aware and Fix Universe State Handling

**Files:**
- Modify: `src/qmt_agent_trader/data/bars.py`
- Modify: `src/qmt_agent_trader/universe/resolver.py`
- Create: `tests/unit/data/test_asset_aware_trade_state.py`
- Modify: `tests/unit/universe/test_resolver.py`

**Interfaces:**
- Every normalized row has `asset_type` equal to `stock` or `etf`.
- ETF rows raise `UNSUPPORTED_ETF_TRADE_STATE_MODEL` until a dedicated ETF source exists.
- Universe resolution consumes validated stock state and never evaluates `bool(pd.NA)`.

- [ ] **Step 1: Write stock/ETF tests**

```python
def test_etf_does_not_reuse_stock_limit_source(tmp_path) -> None:
    lake = DataLake(tmp_path / "lake", tmp_path / "db.duckdb")
    lake.write_parquet(pd.DataFrame([{
        "ts_code": "510300.SH",
        "trade_date": "20240102",
        "open": 3.5,
        "high": 3.6,
        "low": 3.4,
        "close": 3.55,
        "vol": 100.0,
        "amount": 350.0,
    }]), "raw", "tushare/fund_daily")
    with pytest.raises(BacktestDataIntegrityError) as exc:
        load_daily_bars(lake, start="20240102", end="20240102", symbols=["510300.SH"])
    assert exc.value.code == "UNSUPPORTED_ETF_TRADE_STATE_MODEL"
```

Add a stock test asserting `asset_type == "stock"` and non-null state values when valid stock sources exist.

- [ ] **Step 2: Verify failure**

```bash
uv run pytest tests/unit/data/test_asset_aware_trade_state.py -q
```

- [ ] **Step 3: Annotate data source type before concatenation**

Change:

```python
def normalize_tushare_daily(frame: pd.DataFrame, *, asset_type: str | None = None) -> pd.DataFrame:
```

Add `asset_type` to canonical columns. In `load_daily_bars()`, call:

```python
normalize_tushare_daily(stock_raw, asset_type="stock")
normalize_tushare_daily(etf_raw, asset_type="etf")
```

Split rows before state enrichment. If any ETF row is present with `include_trade_state=True`, raise:

```python
BacktestDataIntegrityError(
    code="UNSUPPORTED_ETF_TRADE_STATE_MODEL",
    message="ETF rows cannot use stock-only stk_limit evidence",
    field="trade_state",
    symbols=tuple(sorted(etf_bars["symbol"].astype(str).unique())),
)
```

- [ ] **Step 4: Make UniverseResolver use the validated loader**

Replace the latest-row SQL path with:

```python
bars = load_daily_bars(self.lake, end=as_of_date, include_trade_state=True)
bars = bars[bars["asset_type"].isin(asset_types)].copy()
latest = (
    bars.sort_values(["symbol", "trade_date"], kind="stable")
    .groupby("symbol", as_index=False, group_keys=False)
    .tail(1)
    .reset_index(drop=True)
)
return latest
```

Delete `_apply_fast_st_state()`. Direct `bool(row.get(...))` access must be replaced with validated, non-null direct column access.

- [ ] **Step 5: Add the UniverseResolver regression**

Construct raw daily data without embedded `st`/`suspended`, provide valid `suspend_d`, `stk_limit`, `namechange`, and `stock_basic`, then assert snapshot universe resolution returns the stock rather than raising the ambiguous-NA boolean error.

- [ ] **Step 6: Verify and commit**

```bash
uv run pytest \
  tests/unit/data/test_asset_aware_trade_state.py \
  tests/unit/data/test_trade_state_evidence.py \
  tests/unit/universe/test_resolver.py -q

git add src/qmt_agent_trader/data/bars.py \
        src/qmt_agent_trader/universe/resolver.py \
        tests/unit/data/test_asset_aware_trade_state.py \
        tests/unit/universe/test_resolver.py

git commit -m "fix(data): make trade state asset aware"
```

---

## Task 3: Resolve Strategy Identity and Temporary Semantics Before Universe or Cache Work

**Files:**
- Modify: `src/qmt_agent_trader/agent/tools/strategy_tools.py`
- Create: `tests/unit/agent/test_backtest_pre_cache_identity.py`
- Modify: `tests/unit/strategy/test_backtest_rolling_universe.py`

**Interfaces:**
- Produce `_ResolvedBacktestIntent`.
- Effective strategy ID is top-level ID, otherwise inline spec ID, otherwise temporary factor ID.
- Registry and generated-code guards finish before universe and cache work.
- Temporary factor spec exists before universe resolution.

- [ ] **Step 1: Write pre-cache guard tests**

Test both cases with monkeypatches that raise if cache or universe is reached:

```python
result = strategy_tools._run_backtest({
    "strategy_spec": conflicting_inline_spec.model_dump(mode="json"),
    "start_date": "20240101",
    "end_date": "20240331",
    "symbols": ["000001.SZ"],
}, ToolContext(run_id="inline-saved-conflict"))
assert result["reason"] == "SAVED_STRATEGY_SPEC_MISMATCH"
```

```python
result = strategy_tools._run_backtest({
    "strategy_spec": saved_generated.spec.model_dump(mode="json"),
    "start_date": "20240101",
    "end_date": "20240331",
    "symbols": ["000001.SZ"],
}, ToolContext(run_id="generated-pre-cache"))
assert result["reason"] == "GENERATED_STRATEGY_EXECUTION_NOT_IMPLEMENTED"
```

The second request intentionally omits top-level `strategy_id`.

- [ ] **Step 2: Verify failure**

```bash
uv run pytest tests/unit/agent/test_backtest_pre_cache_identity.py -q
```

- [ ] **Step 3: Add resolved intent**

```python
@dataclass(frozen=True)
class _ResolvedBacktestIntent:
    strategy_id: str
    strategy_spec: StrategySpec
    saved_strategy: SavedStrategy | None
    effective_code_path: str | None
    factor_name: str
    requested_factor_ids: tuple[str, ...]
    strategy_frequency: Literal["daily", "weekly", "monthly"]
```

Implement `_resolve_backtest_intent()` so it:

1. parses inline spec;
2. derives effective ID from top-level or inline spec;
3. loads Registry strategy by effective ID;
4. blocks Registry/inline fingerprint mismatch;
5. creates factor-only temporary spec before universe work;
6. evaluates generated-code and adapter-capability guards;
7. returns one resolved object.

- [ ] **Step 4: Reorder `_run_backtest()`**

Required order:

```text
parse request
resolve Registry identity and generated-code capability
build temporary factor spec when needed
validate StrategySpec/config semantics
resolve universe
build provenance
build cache key
read cache
execute adapter
```

No Registry identity check may remain exclusively after cache lookup.

- [ ] **Step 5: Fix rolling frequency defaults**

Pass `strategy_frequency` explicitly into `_resolve_backtest_universe()`. Compute:

```python
universe_frequency = str(
    input_data.get("universe_rebalance_frequency")
    or strategy_frequency
)
```

Use the same value when constructing broad universe specs and their fingerprints.

- [ ] **Step 6: Verify and commit**

```bash
uv run pytest \
  tests/unit/agent/test_backtest_pre_cache_identity.py \
  tests/unit/agent/test_agent_backtest_config_spec_consistency.py \
  tests/unit/agent/test_saved_generated_strategy_backtest_guard.py \
  tests/unit/strategy/test_backtest_rolling_universe.py -q

git add src/qmt_agent_trader/agent/tools/strategy_tools.py \
        tests/unit/agent/test_backtest_pre_cache_identity.py \
        tests/unit/strategy/test_backtest_rolling_universe.py

git commit -m "fix(agent): resolve strategy identity before cache"
```

---

## Task 4: Replace Partial Cache Keys with a Provenance Manifest

**Files:**
- Create: `src/qmt_agent_trader/persistence/provenance.py`
- Modify: `src/qmt_agent_trader/agent/tools/strategy_tools.py`
- Create: `tests/unit/agent/test_backtest_cache_provenance.py`

**Interfaces:**
- Produce `fingerprint_path_tree(path: Path) -> str | None`.
- Produce `_backtest_provenance_manifest(...) -> dict[str, Any]`.
- Set cache schema to `factor-rank-v3`.

- [ ] **Step 1: Write fingerprint tests**

```python
def test_tree_fingerprint_changes_when_nested_file_changes(tmp_path) -> None:
    root = tmp_path / "dataset"
    root.mkdir()
    part = root / "part-000.parquet"
    part.write_bytes(b"first")
    first = fingerprint_path_tree(root)
    part.write_bytes(b"second-longer")
    second = fingerprint_path_tree(root)
    assert first != second
```

Add cache-key tests that independently change:

- `trade_cal`;
- `namechange`;
- `daily_basic`;
- saved strategy code path contents;
- resolved universe payload.

Each change must produce a different cache key.

- [ ] **Step 2: Implement path-tree fingerprinting**

```python
def fingerprint_path_tree(path: Path) -> str | None:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    files = [path] if path.is_file() else sorted(item for item in path.rglob("*") if item.is_file())
    root = path.parent if path.is_file() else path
    for item in files:
        stat = item.stat()
        digest.update(
            f"{item.relative_to(root).as_posix()}\0{stat.st_size}\0{stat.st_mtime_ns}\n".encode()
        )
    return digest.hexdigest()
```

- [ ] **Step 3: Build complete provenance**

Set:

```python
BACKTEST_CACHE_SCHEMA_VERSION = "factor-rank-v3"
BACKTEST_ENGINE_SEMANTIC_VERSION = "2026-07-opening-state-warmup-v1"
```

Manifest must contain:

```python
{
    "cache_schema_version": BACKTEST_CACHE_SCHEMA_VERSION,
    "engine_semantic_version": BACKTEST_ENGINE_SEMANTIC_VERSION,
    "strategy_spec_fingerprint": ...,
    "saved_strategy": {
        "strategy_id": ...,
        "version": ...,
        "status": ...,
        "spec_fingerprint": ...,
        "code_path": ...,
        "code_fingerprint": ...,
    },
    "factor_fingerprints": ...,
    "dataset_fingerprints": ...,
    "universe_resolution": resolved_universe,
}
```

Dataset set must include market bars, `trade_cal`, `suspend_d`, `stk_limit`, `namechange`, `stock_basic`, and every source selected by required factor fields, including daily-basic, financial, and macro datasets.

- [ ] **Step 4: Use provenance in cache key**

```python
payload = {
    "schema_version": BACKTEST_CACHE_SCHEMA_VERSION,
    "config": config.model_dump(mode="json"),
    "factor_name": factor_name,
    "requested_factor_ids": requested_factor_ids,
    "provenance": provenance,
}
```

Store the manifest in completed payloads and reports. Delete the old fixed four-dataset `_data_fingerprint()`.

- [ ] **Step 5: Verify and commit**

```bash
uv run pytest \
  tests/unit/agent/test_backtest_cache_provenance.py \
  tests/unit/agent/test_backtest_pre_cache_identity.py -q

git add src/qmt_agent_trader/persistence/provenance.py \
        src/qmt_agent_trader/agent/tools/strategy_tools.py \
        tests/unit/agent/test_backtest_cache_provenance.py

git commit -m "fix(cache): fingerprint complete backtest provenance"
```

---

## Task 5: Validate Warm-Up Sessions and Per-Symbol Factor History

**Files:**
- Modify: `src/qmt_agent_trader/data/trading_calendar.py`
- Modify: `src/qmt_agent_trader/strategy/execution_adapter.py`
- Modify: `src/qmt_agent_trader/backtest/research_runner.py`
- Modify: `src/qmt_agent_trader/backtest/research_models.py`
- Create: `tests/unit/backtest/test_warmup_coverage.py`

**Interfaces:**
- Produce `_validate_warmup_panel(...) -> dict[str, object]`.
- Missing whole warm-up sessions raise `MISSING_FACTOR_WARMUP_SESSION`.
- Per-symbol insufficient history is explicit data-quality evidence and excludes that symbol until ready.

- [ ] **Step 1: Write warm-up coverage tests**

```python
def test_missing_whole_warmup_session_fails_closed() -> None:
    panel = pd.DataFrame([
        {"symbol": "000001.SZ", "trade_date": date(2024, 1, 2)},
        {"symbol": "000001.SZ", "trade_date": date(2024, 1, 4)},
    ])
    with pytest.raises(BacktestDataIntegrityError) as exc:
        _validate_warmup_panel(
            panel,
            warmup_dates=(date(2024, 1, 2), date(2024, 1, 3)),
            expected_trade_dates=(date(2024, 1, 4),),
            required_symbols=("000001.SZ",),
            lookback_sessions=2,
        )
    assert exc.value.code == "MISSING_FACTOR_WARMUP_SESSION"
```

```python
def test_insufficient_symbol_history_is_explicit() -> None:
    panel = pd.DataFrame([
        {"symbol": "A", "trade_date": date(2024, 1, 2)},
        {"symbol": "A", "trade_date": date(2024, 1, 3)},
        {"symbol": "B", "trade_date": date(2024, 1, 3)},
        {"symbol": "A", "trade_date": date(2024, 1, 4)},
        {"symbol": "B", "trade_date": date(2024, 1, 4)},
    ])
    quality = _validate_warmup_panel(
        panel,
        warmup_dates=(date(2024, 1, 2), date(2024, 1, 3)),
        expected_trade_dates=(date(2024, 1, 4),),
        required_symbols=("A", "B"),
        lookback_sessions=2,
    )
    assert quality["insufficient_history_by_symbol"] == {
        "B": {"observed_sessions": 1, "required_sessions": 2}
    }
```

- [ ] **Step 2: Validate calendar evidence across warm-up and performance ranges**

After deriving `warmup_dates`, check every natural date from the first warm-up date through requested end exists in normalized calendar states. Missing dates remain `TRADING_CALENDAR_PARTIAL_COVERAGE`.

- [ ] **Step 3: Implement `_validate_warmup_panel()`**

```python
def _validate_warmup_panel(
    panel: pd.DataFrame,
    *,
    warmup_dates: tuple[date, ...],
    expected_trade_dates: tuple[date, ...],
    required_symbols: tuple[str, ...],
    lookback_sessions: int,
) -> dict[str, object]:
    observed_dates = set(panel["trade_date"])
    missing_dates = sorted(set(warmup_dates) - observed_dates)
    if missing_dates:
        raise BacktestDataIntegrityError(
            code="MISSING_FACTOR_WARMUP_SESSION",
            message="factor input panel lacks a required warm-up session",
            field="trade_date",
            details={"missing_dates": [item.isoformat() for item in missing_dates]},
        )
    warmup_set = set(warmup_dates)
    counts = (
        panel[
            panel["trade_date"].isin(warmup_set)
            & panel["symbol"].astype(str).isin(required_symbols)
        ]
        .groupby("symbol")["trade_date"]
        .nunique()
        .to_dict()
    )
    insufficient = {
        symbol: {
            "observed_sessions": int(counts.get(symbol, 0)),
            "required_sessions": lookback_sessions,
        }
        for symbol in required_symbols
        if int(counts.get(symbol, 0)) < lookback_sessions
    }
    return {
        "warmup_session_count": len(warmup_dates),
        "performance_session_count": len(expected_trade_dates),
        "insufficient_history_by_symbol": insufficient,
    }
```

- [ ] **Step 4: Enforce symbol readiness**

Add the insufficient-history mapping to `FactorRankResearchConfig` and `ResearchDataQuality`. In `_prepare_scheduled_signal_frames()`, remove symbols still lacking required history before ranking. If no rows remain, use skip reason `factor_signal_empty_after_history_filter`.

- [ ] **Step 5: Verify and commit**

```bash
uv run pytest \
  tests/unit/backtest/test_warmup_coverage.py \
  tests/unit/strategy/test_backtest_warmup.py \
  tests/unit/backtest/test_research_runner_signal_availability.py \
  tests/unit/backtest/test_research_models.py -q

git add src/qmt_agent_trader/data/trading_calendar.py \
        src/qmt_agent_trader/strategy/execution_adapter.py \
        src/qmt_agent_trader/backtest/research_runner.py \
        src/qmt_agent_trader/backtest/research_models.py \
        tests/unit/backtest/test_warmup_coverage.py

git commit -m "fix(backtest): validate factor warmup coverage"
```

---

## Task 6: Exclude Warm-Up Rows from Diagnostics and Reports

**Files:**
- Modify: `src/qmt_agent_trader/strategy/execution_adapter.py`
- Create: `tests/unit/strategy/test_performance_diagnostic_window.py`

**Interfaces:**
- Produce `_performance_window(frame, expected_dates) -> pd.DataFrame`.
- Diagnostics receive only requested performance dates.

- [ ] **Step 1: Write window tests**

```python
def test_performance_window_excludes_warmup_dates() -> None:
    frame = pd.DataFrame([
        {"symbol": "A", "trade_date": date(2024, 1, 2), "factor_value": 100.0},
        {"symbol": "A", "trade_date": date(2024, 1, 4), "factor_value": 1.0},
        {"symbol": "A", "trade_date": date(2024, 1, 5), "factor_value": 2.0},
    ])
    result = _performance_window(frame, (date(2024, 1, 4), date(2024, 1, 5)))
    assert set(result["trade_date"]) == {date(2024, 1, 4), date(2024, 1, 5)}
```

Add an evidence-level test asserting `factor_report.ic_by_date` contains no pre-start date.

- [ ] **Step 2: Implement and wire the filter**

```python
def _performance_window(frame: pd.DataFrame, expected_dates: tuple[date, ...]) -> pd.DataFrame:
    if frame.empty or "trade_date" not in frame.columns:
        return frame.copy()
    dates = pd.to_datetime(frame["trade_date"], errors="coerce").dt.date
    return frame.loc[dates.isin(set(expected_dates))].copy()
```

Before `_diagnostic_evidence()`:

```python
performance_factor_frame = _performance_window(runner.factor_frame, session_window.expected_dates)
performance_bars = _performance_window(runner.bars, session_window.expected_dates)
```

Pass these frames, not the warm-up-inclusive frames. Add report metadata for diagnostic start/end and excluded row count.

- [ ] **Step 3: Verify and commit**

```bash
uv run pytest \
  tests/unit/strategy/test_performance_diagnostic_window.py \
  tests/unit/strategy/test_backtest_diagnostic_wiring.py \
  tests/unit/strategy/test_backtest_report_schema.py -q

git add src/qmt_agent_trader/strategy/execution_adapter.py \
        tests/unit/strategy/test_performance_diagnostic_window.py

git commit -m "fix(diagnostics): exclude factor warmup rows"
```

---

## Task 7: Reject Remaining ASOF and Universe Source Ambiguity

**Files:**
- Modify: `src/qmt_agent_trader/factors/input_panel.py`
- Modify: `src/qmt_agent_trader/universe/resolver.py`
- Create: `tests/unit/factors/test_asof_ambiguity.py`
- Modify: `tests/unit/universe/test_resolver.py`

**Interfaces:**
- Duplicate ASOF visible keys raise `DUPLICATE_ASOF_VISIBLE_KEY`.
- Universe raw symbol-date duplicates fail before latest-row selection.

- [ ] **Step 1: Write ASOF ambiguity tests**

```python
def test_symbol_asof_duplicate_visible_key_fails_closed() -> None:
    panel = pd.DataFrame([{"symbol": "A", "trade_date": date(2024, 1, 5)}])
    source = pd.DataFrame([
        {"symbol": "A", "visible_date": date(2024, 1, 4), "roe": 0.10},
        {"symbol": "A", "visible_date": date(2024, 1, 4), "roe": 0.12},
    ])
    with pytest.raises(BacktestDataIntegrityError) as exc:
        _join_symbol_asof(panel, source, "roe")
    assert exc.value.code == "DUPLICATE_ASOF_VISIBLE_KEY"
```

Add the equivalent marketwide test for duplicate `visible_date`.

- [ ] **Step 2: Replace `keep="last"` with validation**

Before each ASOF merge:

```python
duplicate = right.duplicated(["symbol", "visible_date"], keep=False)
if duplicate.any():
    raise BacktestDataIntegrityError(
        code="DUPLICATE_ASOF_VISIBLE_KEY",
        message="ASOF source has multiple values for one visible key",
        field=field,
    )
```

For marketwide sources, key is `visible_date`. Remove both `drop_duplicates(..., keep="last")` calls.

- [ ] **Step 3: Validate universe raw identities before latest selection**

Use the validated `load_daily_bars()` path from Task 2. Remove SQL `row_number()` latest selection that can hide duplicate symbol-date rows. In `_market_cap_asof()`, validate `daily_basic` symbol-date uniqueness before sorting and selecting latest rows.

- [ ] **Step 4: Verify and commit**

```bash
uv run pytest \
  tests/unit/factors/test_asof_ambiguity.py \
  tests/unit/factors/test_input_panel.py \
  tests/unit/universe/test_resolver.py \
  tests/unit/data/test_data_integrity.py -q

git add src/qmt_agent_trader/factors/input_panel.py \
        src/qmt_agent_trader/universe/resolver.py \
        tests/unit/factors/test_asof_ambiguity.py \
        tests/unit/universe/test_resolver.py

git commit -m "fix(data): reject remaining source ambiguity"
```

---

## Task 8: Make the Runner Require Prevalidated Canonical Inputs

**Files:**
- Modify: `src/qmt_agent_trader/backtest/research_runner.py`
- Create: `tests/unit/backtest/test_runner_input_contract.py`
- Modify: runner test fixtures under `tests/unit/backtest/`

**Interfaces:**
- Missing execution-state fields raise `MISSING_EXECUTION_STATE_COLUMNS`.
- Null execution-state fields raise `UNKNOWN_EXECUTION_STATE`.
- Missing numeric/factor inputs are not synthesized.

- [ ] **Step 1: Write strict contract tests**

```python
def canonical_row() -> dict[str, object]:
    return {
        "symbol": "000001.SZ",
        "trade_date": date(2024, 1, 2),
        "open": 10.0,
        "high": 10.2,
        "low": 9.8,
        "close": 10.1,
        "volume": 100.0,
        "amount": 1_000.0,
        "turnover": 0.01,
        "suspended": False,
        "st": False,
        "limit_up_at_open": False,
        "limit_down_at_open": False,
    }
```

Test missing `suspended`, null `limit_up_at_open`, and missing `turnover`.

- [ ] **Step 2: Replace fail-open `_prepare_bars()`**

```python
_REQUIRED_CANONICAL_BAR_COLUMNS = {
    "symbol", "trade_date", "open", "high", "low", "close",
    "volume", "amount", "turnover", "suspended", "st",
    "limit_up_at_open", "limit_down_at_open",
}
_EXECUTION_STATE_COLUMNS = {
    "suspended", "st", "limit_up_at_open", "limit_down_at_open",
}
```

Raise a typed integrity error for missing columns. Reject null execution state. Coerce existing numeric columns, but never create absent numeric or boolean columns. Delete all `0.0` and `False` fallback assignments.

- [ ] **Step 3: Update fixtures and verify**

Update legacy unit fixtures to provide the complete canonical row. Do not restore fallback behavior to satisfy them.

```bash
uv run pytest tests/unit/backtest -q

git add src/qmt_agent_trader/backtest/research_runner.py \
        tests/unit/backtest

git commit -m "fix(backtest): require canonical runner inputs"
```

---

## Task 9: Update Documentation and Run Final Verification

**Files:**
- Modify: `docs/backtest/factor-rank-adapter.md`

- [ ] **Step 1: Document final contracts**

Document:

1. opening-only limit-state semantics;
2. strict source schema and positive finite limit prices;
3. stock/ETF separation and `UNSUPPORTED_ETF_TRADE_STATE_MODEL`;
4. effective strategy identity before universe/cache work;
5. cache schema `factor-rank-v3` and complete provenance;
6. warm-up session and symbol-history validation;
7. exclusion of warm-up rows from all diagnostics;
8. ASOF and universe ambiguity errors;
9. strict runner input schema.

- [ ] **Step 2: Run focused tests**

```bash
uv run pytest \
  tests/unit/data/test_opening_trade_state.py \
  tests/unit/data/test_asset_aware_trade_state.py \
  tests/unit/agent/test_backtest_pre_cache_identity.py \
  tests/unit/agent/test_backtest_cache_provenance.py \
  tests/unit/backtest/test_warmup_coverage.py \
  tests/unit/strategy/test_performance_diagnostic_window.py \
  tests/unit/factors/test_asof_ambiguity.py \
  tests/unit/backtest/test_runner_input_contract.py \
  tests/unit/universe/test_resolver.py -q
```

Expected: PASS.

- [ ] **Step 3: Run affected suites**

```bash
uv run pytest \
  tests/unit/backtest \
  tests/unit/data \
  tests/unit/factors \
  tests/unit/strategy \
  tests/unit/universe \
  tests/unit/agent/test_agent_backtest_config_spec_consistency.py \
  tests/unit/agent/test_backtest_integrity_error_boundary.py \
  tests/unit/agent/test_saved_generated_strategy_backtest_guard.py \
  tests/unit/agent/test_backtest_pre_cache_identity.py \
  tests/integration/test_factor_rank_backtest_correctness.py -q
```

Expected: PASS.

- [ ] **Step 4: Inspect forbidden fallbacks and ambiguity**

```bash
rg -n "drop_duplicates|row_number\(\)|fillna\(False\)|get\(\"(suspended|st|limit_up|limit_down|limit_up_at_open|limit_down_at_open)\"" \
  src/qmt_agent_trader/data \
  src/qmt_agent_trader/factors/input_panel.py \
  src/qmt_agent_trader/universe/resolver.py \
  src/qmt_agent_trader/backtest/research_runner.py
```

Every remaining occurrence must be presentation-only, occur after an explicit uniqueness check, or use a complete documented business tie-break. No execution decision may default unknown state to `False`.

- [ ] **Step 5: Verify execution ordering**

```bash
rg -n "_resolve_backtest_intent|_resolve_backtest_universe|_get_cached_backtest" \
  src/qmt_agent_trader/agent/tools/strategy_tools.py
```

Confirm order is identity, temporary spec, universe, provenance, cache, execution.

- [ ] **Step 6: Check broad exception handling**

```bash
rg -n "except Exception" \
  src/qmt_agent_trader/backtest \
  src/qmt_agent_trader/data/trade_state.py \
  src/qmt_agent_trader/data/bars.py \
  src/qmt_agent_trader/factors/input_panel.py \
  src/qmt_agent_trader/strategy/execution_adapter.py \
  src/qmt_agent_trader/agent/tools/strategy_tools.py
```

No broad catch may normalize execution, source validation, Registry identity, provenance, or warm-up failures.

- [ ] **Step 7: Run repository gate and commit docs**

```bash
make check

git add docs/backtest/factor-rank-adapter.md
git commit -m "docs(backtest): document final correctness contracts"
```

Expected: `make check` exits with code `0`.

---

# Final Acceptance Checklist

## Opening execution correctness

- [ ] Opening eligibility uses only opening-time information.
- [ ] Closing limit prices cannot block an opening trade.
- [ ] Limit prices are finite, positive, and key-complete.
- [ ] Unknown execution-state values never become `False`.

## Asset-aware evidence

- [ ] Every normalized row has `asset_type`.
- [ ] Stock state uses stock sources only.
- [ ] ETF rows never reuse `stk_limit`.
- [ ] Unsupported ETF state returns `UNSUPPORTED_ETF_TRADE_STATE_MODEL`.
- [ ] Universe resolution never calls `bool(pd.NA)`.

## Strategy identity and cache

- [ ] Effective strategy ID includes IDs supplied only inside inline specs.
- [ ] Registry conflict and generated-code guards run before universe/cache work.
- [ ] Temporary factor strategy exists before universe resolution.
- [ ] Rolling-universe default cadence equals authoritative strategy cadence.
- [ ] Cache schema is `factor-rank-v3`.
- [ ] All effective datasets, strategy state, factors, universe, and engine semantics are fingerprinted.

## Warm-up correctness

- [ ] Warm-up calendar evidence is complete.
- [ ] Every required warm-up session exists in the panel.
- [ ] Insufficient symbol history is explicit and filtered before ranking.
- [ ] Warm-up rows never enter trades, equity, metrics, IC, coverage, or walk-forward evidence.

## Source ambiguity and runner contract

- [ ] Exact, ASOF, trade-state, and universe identity duplicates fail closed.
- [ ] No business identity is resolved by storage order.
- [ ] Direct runner calls require complete canonical inputs.
- [ ] Missing or null execution state raises typed errors.
- [ ] Missing numeric/factor fields are not synthesized.

## Safety and verification

- [ ] Integrity errors create no completed report or successful cache entry.
- [ ] Unexpected software exceptions propagate.
- [ ] `research_only=True`.
- [ ] `live_trading_allowed=False`.
- [ ] Focused tests pass.
- [ ] Affected suites pass.
- [ ] Existing factor-rank integration test passes.
- [ ] `make check` passes.
- [ ] Documentation matches implementation.

## Explicitly Out of Scope

- Implementing a dedicated ETF limit-price data source.
- Historical extreme-drawdown replay.
- A new full DataLake-to-Agent integration suite.
- GitHub Actions creation or modification.
- Process-isolated execution of generated strategy Python.

## Expected Merge Decision

Keep `REQUEST_CHANGES` until every acceptance item and local verification command passes. Then perform one final static branch review before merging.
