# Factor-Rank Session and Universe Final Repairs Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the remaining session-alignment and universe-integrity gaps so snapshot and rolling universes use one authoritative effective market session, mixed-asset coverage is complete, liquidity ranking cannot admit incomplete windows, index evidence fails closed, and direct profiling calls use the new strategy identity contract.

**Architecture:** Compute the effective market session once at the universe boundary and pass it explicitly through every point-in-time lookup. Keep `trade_cal` authoritative and require complete natural-date evidence between the selected open session and the request boundary. Make asset coverage, liquidity observation coverage, and index membership evidence explicit typed gates before ranking or backtest execution.

**Tech Stack:** Python 3.11+, pandas, Pydantic v2, pytest, existing `DataLake`, `UniverseResolver`, `StrategyBacktestConfig`, `StrategySpec`, Ruff, mypy, and `uv`.

## Global Constraints

- Target branch: `codex/factor-rank-backtest-correctness`; continue from current head `c41c6c1c2054a3be29d58734b70b74808f1c4610`.
- Save this plan as `docs/superpowers/plans/2026-07-17-factor-rank-session-universe-final-repairs.md`.
- Use an isolated worktree at execution time.
- Follow TDD: add one focused failing regression, run it, implement the smallest correct change, rerun, and commit.
- Use one focused commit per task.
- `trade_cal` is the only authority for effective snapshot sessions and rolling rebalance dates.
- A requested closed date may resolve to the previous open session only when every natural date between that session and the request boundary has calendar evidence.
- All PIT metadata, index membership, market-cap lookup, liquidity windows, and diagnostics must use the same effective market session as the loaded bars.
- A mixed universe requiring `stock` and `etf` must have exact-session bars for both asset types.
- A 20-session liquidity field is eligible for filtering or ranking only with exactly 20 official sessions and 20 non-null field observations.
- An index code is resolved only when at least one valid normalized member exists at the target effective session.
- Non-empty malformed index member identifiers and PIT dates fail closed.
- Preserve `research_only=True` and `live_trading_allowed=False`.
- Integrity failures create no completed report and no successful cache entry.
- Unexpected programming exceptions propagate.
- Do not add runtime dependencies.
- Do not add or modify GitHub Actions.
- Do not reproduce the historical extreme-drawdown run.
- Do not create a new full DataLake-to-Agent integration suite.
- Bump the backtest engine semantic version because universe membership semantics change.
- Do not weaken the existing strict runner, trade-state, Registry identity, warm-up, or source-ambiguity contracts.

---

## File Responsibility Map

### Existing files to modify

- `src/qmt_agent_trader/data/trading_calendar.py` — validate continuous natural-date calendar evidence for previous-open-session resolution.
- `src/qmt_agent_trader/universe/resolver.py` — resolve one effective session, propagate it through all PIT paths, require per-asset exact-session bars, reject incomplete ranking metrics, and require non-empty index evidence.
- `src/qmt_agent_trader/universe/pit_metadata.py` — validate and normalize index member identifiers and omit empty memberships from successful evidence maps.
- `src/qmt_agent_trader/agent/tools/strategy_tools.py` — bump the engine semantic version to invalidate old successful-result cache entries.
- `scripts/profile_research_tools.py` — build a valid ad-hoc `StrategySpec` and `StrategyBacktestConfig`.
- `docs/backtest/factor-rank-adapter.md` — document effective-session alignment, per-asset coverage, ranking completeness, and index evidence.

### Existing tests to modify

- `tests/unit/data/test_trading_calendar.py`
- `tests/unit/universe/test_exact_session_resolution.py`
- `tests/unit/universe/test_resolver.py`
- `tests/unit/universe/test_index_membership_asof.py`
- `tests/unit/universe/test_universe_resolver_rolling.py`

### New test file

- `tests/unit/scripts/test_profile_research_tools.py`

---

# Task 1: Resolve One Effective Market Session and Use It Everywhere

**Files:**
- Modify: `src/qmt_agent_trader/universe/resolver.py`
- Modify: `tests/unit/universe/test_exact_session_resolution.py`
- Modify: `tests/unit/universe/test_index_membership_asof.py`

**Interfaces:**
- Produces `_ResolvedUniverseSession`.
- Produces `_resolve_effective_session(lake: DataLake, requested_as_of: str) -> _ResolvedUniverseSession`.
- `_load_recent_bars()` consumes an already resolved `effective_date: date`.
- `_select_candidates()`, `_attach_metrics()`, `_avg_20d_metrics()`, `_market_cap_asof()`, and `_index_constituents()` all consume the same `effective_date`.
- Diagnostics expose both `requested_as_of_date` and `effective_market_session`.

- [ ] **Step 1: Add the closed-boundary PIT regression**

Append to `tests/unit/universe/test_exact_session_resolution.py`:

```python
from datetime import date

from qmt_agent_trader.universe import resolver as resolver_module


def test_closed_boundary_uses_previous_open_session_for_all_pit_inputs(
    tmp_path,
    monkeypatch,
) -> None:
    lake = DataLake(tmp_path / "lake", tmp_path / "research.duckdb")
    lake.write_parquet(
        pd.DataFrame(
            [
                {"exchange": "SSE", "cal_date": "20240105", "is_open": 1},
                {"exchange": "SZSE", "cal_date": "20240105", "is_open": 1},
                {"exchange": "SSE", "cal_date": "20240106", "is_open": 0},
                {"exchange": "SZSE", "cal_date": "20240106", "is_open": 0},
                {"exchange": "SSE", "cal_date": "20240107", "is_open": 0},
                {"exchange": "SZSE", "cal_date": "20240107", "is_open": 0},
            ]
        ),
        "raw",
        "tushare/trade_cal",
    )
    observed: dict[str, object] = {}
    resolver = UniverseResolver(lake)
    spec = UniverseSpec.model_validate(
        {
            "universe_id": "closed-boundary",
            "name": "Closed boundary",
            "source": "user_defined",
            "asset_types": ["stock"],
            "selection": {
                "mode": "index_constituents",
                "index_codes": ["000300.SH"],
            },
            "filters": {"min_listed_days": 0},
        }
    )

    def fake_bars(effective_date, _asset_types):
        observed["bars_date"] = effective_date
        return pd.DataFrame(
            [
                {
                    "symbol": "000001.SZ",
                    "trade_date": effective_date,
                    "asset_type": "stock",
                    "st": False,
                    "suspended": False,
                    "volume": 100.0,
                    "amount": 1000.0,
                }
            ]
        )

    def fake_index(_codes, effective_date):
        observed["index_date"] = effective_date
        return ["000001.SZ"]

    def fake_metrics(frame, _spec, *, effective_date):
        observed["metrics_date"] = effective_date
        return frame

    monkeypatch.setattr(resolver, "_load_recent_bars", fake_bars)
    monkeypatch.setattr(
        resolver,
        "_stock_basic",
        lambda: pd.DataFrame(
            [
                {
                    "ts_code": "000001.SZ",
                    "name": "Fixture",
                    "list_date": "20000101",
                    "delist_date": None,
                }
            ]
        ),
    )
    monkeypatch.setattr(resolver, "_index_constituents", fake_index)
    monkeypatch.setattr(resolver, "_attach_metrics", fake_metrics)

    result = resolver.build(
        spec,
        mode="snapshot",
        as_of_date="20240107",
    )

    assert result["status"] == "OK"
    assert result["symbols"] == ["000001.SZ"]
    assert observed == {
        "bars_date": date(2024, 1, 5),
        "index_date": date(2024, 1, 5),
        "metrics_date": date(2024, 1, 5),
    }
    diagnostics = result["metadata"]["diagnostics"]
    assert diagnostics["requested_as_of_date"] == "20240107"
    assert diagnostics["effective_market_session"] == "20240105"
```

- [ ] **Step 2: Run the new test**

```bash
uv run pytest \
  tests/unit/universe/test_exact_session_resolution.py::test_closed_boundary_uses_previous_open_session_for_all_pit_inputs -q
```

Expected: FAIL because `_resolve_for_date()` currently mixes the requested date with the effective bar date.

- [ ] **Step 3: Add the resolved-session value object**

In `src/qmt_agent_trader/universe/resolver.py`, add:

```python
from dataclasses import dataclass


@dataclass(frozen=True)
class _ResolvedUniverseSession:
    requested_as_of: date
    effective_date: date

    @property
    def requested_key(self) -> str:
        return f"{self.requested_as_of:%Y%m%d}"

    @property
    def effective_key(self) -> str:
        return f"{self.effective_date:%Y%m%d}"


def _resolve_effective_session(
    lake: DataLake,
    requested_as_of: str,
) -> _ResolvedUniverseSession:
    requested = _parse_date(requested_as_of)
    effective = latest_open_session_on_or_before(
        lake,
        as_of=requested,
    )
    return _ResolvedUniverseSession(
        requested_as_of=requested,
        effective_date=effective,
    )
```

- [ ] **Step 4: Resolve the session once in `_resolve_for_date()`**

Replace the beginning of `_resolve_for_date()` with:

```python
session = _resolve_effective_session(
    self.lake,
    as_of_date,
)
recent = self._load_recent_bars(
    session.effective_date,
    spec.asset_types,
)
stock_basic = security_master_asof(
    self._stock_basic(),
    session.effective_date,
)
require_historical_classification_support(
    selection_mode=spec.selection.mode,
    as_of_date=session.effective_date,
    classification_frame=None,
)
candidates = self._select_candidates(
    spec,
    recent,
    stock_basic,
    effective_date=session.effective_date,
)
candidate_count = len(candidates)
candidates = self._attach_metrics(
    candidates,
    spec,
    effective_date=session.effective_date,
)
```

Replace each exclusion call with:

```python
reason = self._exclusion_reason(
    spec,
    row,
    as_of_date=session.effective_key,
)
```

Remove the second `latest_open_session_on_or_before()` call at the end of `_resolve_for_date()` and set:

```python
diagnostics["requested_as_of_date"] = session.requested_key
diagnostics["effective_market_session"] = session.effective_key
```

- [ ] **Step 5: Change helper signatures to accept the effective date**

Use these exact signatures:

```python
def _load_recent_bars(
    self,
    effective_date: date,
    asset_types: Sequence[str],
) -> pd.DataFrame:
```

```python
def _select_candidates(
    self,
    spec: UniverseSpec,
    recent: pd.DataFrame,
    stock_basic: pd.DataFrame,
    *,
    effective_date: date,
) -> pd.DataFrame:
```

```python
def _attach_metrics(
    self,
    candidates: pd.DataFrame,
    spec: UniverseSpec,
    *,
    effective_date: date,
) -> pd.DataFrame:
```

```python
def _avg_20d_metrics(
    self,
    effective_date: date,
    asset_types: Sequence[str],
) -> pd.DataFrame:
```

```python
def _market_cap_asof(
    self,
    effective_date: date,
) -> pd.DataFrame:
```

```python
def _index_constituents(
    self,
    index_codes: list[str],
    effective_date: date,
) -> list[str]:
```

Inside those methods, derive keys only from `effective_date`:

```python
key = f"{effective_date:%Y%m%d}"
```

Remove all independent effective-session resolution from these helpers.

- [ ] **Step 6: Add an index weekend-boundary regression**

Append to `tests/unit/universe/test_index_membership_asof.py`:

```python
def test_index_membership_uses_effective_market_session_not_closed_boundary(
    tmp_path,
) -> None:
    lake = DataLake(tmp_path / "lake", tmp_path / "research.duckdb")
    lake.write_parquet(
        pd.DataFrame(
            [
                {
                    "index_code": "000300.SH",
                    "con_code": "000001.SZ",
                    "trade_date": "20240105",
                },
                {
                    "index_code": "000300.SH",
                    "con_code": "000002.SZ",
                    "trade_date": "20240108",
                },
            ]
        ),
        "raw",
        "tushare/index_weight",
    )

    observed = UniverseResolver(lake)._index_constituents(
        ["000300.SH"],
        date(2024, 1, 5),
    )

    assert observed == ["000001.SZ"]
```

- [ ] **Step 7: Run focused tests**

```bash
uv run pytest \
  tests/unit/universe/test_exact_session_resolution.py \
  tests/unit/universe/test_index_membership_asof.py -q
```

Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add src/qmt_agent_trader/universe/resolver.py \
        tests/unit/universe/test_exact_session_resolution.py \
        tests/unit/universe/test_index_membership_asof.py
git commit -m "fix(universe): align PIT inputs to effective session"
```

---

# Task 2: Require Continuous Calendar Evidence for Previous-Open Resolution

**Files:**
- Modify: `src/qmt_agent_trader/data/trading_calendar.py`
- Modify: `tests/unit/data/test_trading_calendar.py`

**Interfaces:**
- `latest_open_session_on_or_before()` validates every natural date from the selected open candidate through the requested boundary.
- Missing intermediate evidence raises `TRADING_CALENDAR_PARTIAL_COVERAGE`.
- A fully evidenced weekend or holiday boundary remains valid.

- [ ] **Step 1: Write the intermediate-gap regression**

Append to `tests/unit/data/test_trading_calendar.py`:

```python
def test_latest_open_session_rejects_intermediate_calendar_gap(tmp_path) -> None:
    lake = data_lake(tmp_path)
    lake.write_parquet(
        pd.DataFrame(
            [
                {"exchange": "SSE", "cal_date": "20240101", "is_open": 1},
                {"exchange": "SZSE", "cal_date": "20240101", "is_open": 1},
                {"exchange": "SSE", "cal_date": "20240103", "is_open": 0},
                {"exchange": "SZSE", "cal_date": "20240103", "is_open": 0},
            ]
        ),
        "raw",
        "tushare/trade_cal",
    )

    with pytest.raises(BacktestDataIntegrityError) as exc_info:
        latest_open_session_on_or_before(
            lake,
            as_of="20240103",
        )

    assert exc_info.value.code == "TRADING_CALENDAR_PARTIAL_COVERAGE"
    assert exc_info.value.details["missing_dates"] == ["2024-01-02"]
```

- [ ] **Step 2: Run the regression**

```bash
uv run pytest \
  tests/unit/data/test_trading_calendar.py::test_latest_open_session_rejects_intermediate_calendar_gap -q
```

Expected: FAIL because the function currently checks only the boundary date.

- [ ] **Step 3: Validate the candidate-to-boundary range**

Replace the final `return max(candidates)` with:

```python
candidate = max(candidates)
required_dates = _natural_dates(
    candidate,
    boundary,
)
missing_dates = [
    day
    for day in required_dates
    if day not in states
]
if missing_dates:
    raise BacktestDataIntegrityError(
        code="TRADING_CALENDAR_PARTIAL_COVERAGE",
        message=(
            "trade calendar lacks continuous evidence between "
            "the previous open session and requested boundary"
        ),
        field="trade_cal",
        details={
            "candidate_open_session": candidate.isoformat(),
            "as_of": boundary.isoformat(),
            "missing_dates": [
                day.isoformat()
                for day in missing_dates
            ],
        },
    )
return candidate
```

- [ ] **Step 4: Add the fully evidenced weekend regression**

Append:

```python
def test_latest_open_session_accepts_continuously_evidenced_weekend(
    tmp_path,
) -> None:
    lake = data_lake(tmp_path)
    lake.write_parquet(
        pd.DataFrame(
            [
                {"exchange": "SSE", "cal_date": "20240105", "is_open": 1},
                {"exchange": "SZSE", "cal_date": "20240105", "is_open": 1},
                {"exchange": "SSE", "cal_date": "20240106", "is_open": 0},
                {"exchange": "SZSE", "cal_date": "20240106", "is_open": 0},
                {"exchange": "SSE", "cal_date": "20240107", "is_open": 0},
                {"exchange": "SZSE", "cal_date": "20240107", "is_open": 0},
            ]
        ),
        "raw",
        "tushare/trade_cal",
    )

    assert latest_open_session_on_or_before(
        lake,
        as_of="20240107",
    ) == date(2024, 1, 5)
```

- [ ] **Step 5: Run the calendar suite**

```bash
uv run pytest tests/unit/data/test_trading_calendar.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/qmt_agent_trader/data/trading_calendar.py \
        tests/unit/data/test_trading_calendar.py
git commit -m "fix(calendar): require continuous session evidence"
```

---

# Task 3: Require Exact-Session Coverage for Every Requested Asset Type

**Files:**
- Modify: `src/qmt_agent_trader/universe/resolver.py`
- Modify: `tests/unit/universe/test_exact_session_resolution.py`

**Interfaces:**
- `_load_recent_bars()` compares requested and observed asset types.
- Missing requested asset types raise `UNIVERSE_MARKET_SESSION_NOT_READY`.
- Error details include requested, observed, and missing asset types.

- [ ] **Step 1: Write the mixed-universe coverage regression**

Append to `tests/unit/universe/test_exact_session_resolution.py`:

```python
def test_mixed_universe_requires_each_requested_asset_type(
    tmp_path,
    monkeypatch,
) -> None:
    resolver = UniverseResolver(
        DataLake(tmp_path / "lake", tmp_path / "research.duckdb")
    )
    monkeypatch.setattr(
        resolver_module,
        "load_daily_bars",
        lambda *_args, **_kwargs: pd.DataFrame(
            [
                {
                    "symbol": "000001.SZ",
                    "trade_date": date(2024, 1, 2),
                    "asset_type": "stock",
                }
            ]
        ),
    )

    with pytest.raises(BacktestUniverseIntegrityError) as exc_info:
        resolver._load_recent_bars(
            date(2024, 1, 2),
            ["stock", "etf"],
        )

    assert exc_info.value.code == "UNIVERSE_MARKET_SESSION_NOT_READY"
    assert exc_info.value.details == {
        "requested_asset_types": ["etf", "stock"],
        "observed_asset_types": ["stock"],
        "missing_asset_types": ["etf"],
    }
```

- [ ] **Step 2: Run the regression**

```bash
uv run pytest \
  tests/unit/universe/test_exact_session_resolution.py::test_mixed_universe_requires_each_requested_asset_type -q
```

Expected: FAIL because `_load_recent_bars()` currently checks only `exact.empty`.

- [ ] **Step 3: Add the per-asset coverage gate**

After creating `exact`, use:

```python
requested_asset_types = {
    str(item)
    for item in asset_types
}
observed_asset_types = {
    str(item)
    for item in exact["asset_type"].dropna().astype(str)
}
missing_asset_types = requested_asset_types.difference(
    observed_asset_types
)
if exact.empty or missing_asset_types:
    raise BacktestUniverseIntegrityError(
        code="UNIVERSE_MARKET_SESSION_NOT_READY",
        message=(
            "official open session lacks market bars for "
            "one or more requested asset types"
        ),
        trade_date=effective_date.isoformat(),
        field="daily_bars",
        details={
            "requested_asset_types": sorted(
                requested_asset_types
            ),
            "observed_asset_types": sorted(
                observed_asset_types
            ),
            "missing_asset_types": sorted(
                missing_asset_types
            ),
        },
    )
```

Delete the old `if exact.empty` block.

- [ ] **Step 4: Add a successful mixed-session regression**

```python
def test_mixed_universe_accepts_stock_and_etf_rows(
    tmp_path,
    monkeypatch,
) -> None:
    resolver = UniverseResolver(
        DataLake(tmp_path / "lake", tmp_path / "research.duckdb")
    )
    expected = pd.DataFrame(
        [
            {
                "symbol": "000001.SZ",
                "trade_date": date(2024, 1, 2),
                "asset_type": "stock",
            },
            {
                "symbol": "510300.SH",
                "trade_date": date(2024, 1, 2),
                "asset_type": "etf",
            },
        ]
    )
    monkeypatch.setattr(
        resolver_module,
        "load_daily_bars",
        lambda *_args, **_kwargs: expected.copy(),
    )

    observed = resolver._load_recent_bars(
        date(2024, 1, 2),
        ["stock", "etf"],
    )

    assert observed["asset_type"].tolist() == ["stock", "etf"]
```

- [ ] **Step 5: Run focused tests**

```bash
uv run pytest \
  tests/unit/universe/test_exact_session_resolution.py \
  tests/unit/universe/test_resolver.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/qmt_agent_trader/universe/resolver.py \
        tests/unit/universe/test_exact_session_resolution.py
git commit -m "fix(universe): require per-asset session coverage"
```

---

# Task 4: Exclude Incomplete Liquidity Metrics Before Ranking

**Files:**
- Modify: `src/qmt_agent_trader/universe/resolver.py`
- Modify: `tests/unit/universe/test_resolver.py`

**Interfaces:**
- Produces `_ranking_eligible_rows(frame: pd.DataFrame, field: str) -> pd.DataFrame`.
- `avg_amount_20d` ranking requires `amount_observation_count == 20`.
- `avg_volume_20d` ranking requires `volume_observation_count == 20`.
- `top_n` never fills from invalid liquidity rows.

- [ ] **Step 1: Write the invalid-fill regression**

Append to `tests/unit/universe/test_resolver.py`:

```python
def test_liquidity_ranking_does_not_fill_top_n_with_incomplete_window(
    tmp_path,
) -> None:
    spec = UniverseSpec.model_validate(
        {
            "universe_id": "ranked-liquidity",
            "name": "Ranked liquidity",
            "source": "user_defined",
            "asset_types": ["stock"],
            "selection": {"mode": "all"},
            "ranking": {
                "field": "avg_amount_20d",
                "ascending": False,
                "top_n": 2,
            },
            "filters": {"min_listed_days": 0},
        }
    )
    frame = pd.DataFrame(
        [
            {
                "symbol": "000001.SZ",
                "avg_amount_20d": 1000.0,
                "amount_observation_count": 20,
            },
            {
                "symbol": "000002.SZ",
                "avg_amount_20d": pd.NA,
                "amount_observation_count": 19,
            },
        ]
    )
    resolver = UniverseResolver(
        DataLake(tmp_path / "lake", tmp_path / "research.duckdb")
    )

    ranked = resolver._apply_ranking(frame, spec)

    assert ranked["symbol"].tolist() == ["000001.SZ"]
```

- [ ] **Step 2: Add the volume regression**

```python
def test_volume_ranking_requires_twenty_non_null_observations(
    tmp_path,
) -> None:
    spec = UniverseSpec.model_validate(
        {
            "universe_id": "ranked-volume",
            "name": "Ranked volume",
            "source": "user_defined",
            "asset_types": ["stock"],
            "selection": {"mode": "all"},
            "ranking": {
                "field": "avg_volume_20d",
                "ascending": False,
                "top_n": 5,
            },
            "filters": {"min_listed_days": 0},
        }
    )
    frame = pd.DataFrame(
        [
            {
                "symbol": "000001.SZ",
                "avg_volume_20d": 100.0,
                "volume_observation_count": 20,
            },
            {
                "symbol": "000002.SZ",
                "avg_volume_20d": 200.0,
                "volume_observation_count": 19,
            },
        ]
    )
    resolver = UniverseResolver(
        DataLake(tmp_path / "lake", tmp_path / "research.duckdb")
    )

    ranked = resolver._apply_ranking(frame, spec)

    assert ranked["symbol"].tolist() == ["000001.SZ"]
```

- [ ] **Step 3: Run the regressions**

```bash
uv run pytest \
  tests/unit/universe/test_resolver.py::test_liquidity_ranking_does_not_fill_top_n_with_incomplete_window \
  tests/unit/universe/test_resolver.py::test_volume_ranking_requires_twenty_non_null_observations -q
```

Expected: FAIL because incomplete rows remain after sorting.

- [ ] **Step 4: Add ranking eligibility**

In `resolver.py`, add:

```python
def _ranking_eligible_rows(
    frame: pd.DataFrame,
    field: str,
) -> pd.DataFrame:
    if field not in frame.columns:
        return frame.iloc[0:0].copy()
    eligible = frame.dropna(
        subset=[field]
    ).copy()
    count_field = {
        "avg_amount_20d": "amount_observation_count",
        "avg_volume_20d": "volume_observation_count",
    }.get(field)
    if count_field is None:
        return eligible
    if count_field not in eligible.columns:
        return eligible.iloc[0:0].copy()
    counts = pd.to_numeric(
        eligible[count_field],
        errors="coerce",
    )
    return eligible.loc[
        counts.eq(LIQUIDITY_WINDOW_SESSIONS)
    ].copy()
```

Replace `_apply_ranking()` with:

```python
def _apply_ranking(
    self,
    frame: pd.DataFrame,
    spec: UniverseSpec,
) -> pd.DataFrame:
    ranking = spec.ranking
    if frame.empty or ranking is None:
        return frame
    if ranking.field not in frame.columns or "symbol" not in frame.columns:
        return frame.iloc[0:0].copy()
    eligible = _ranking_eligible_rows(
        frame,
        ranking.field,
    )
    ranked = eligible.sort_values(
        [ranking.field, "symbol"],
        ascending=[ranking.ascending, True],
        na_position="last",
        kind="stable",
    )
    if ranking.top_n is not None:
        ranked = ranked.head(ranking.top_n)
    return ranked
```

- [ ] **Step 5: Add ranking diagnostics**

Before and after ranking in `_resolve_for_date()`:

```python
pre_ranking_count = len(selected_frame)
selected_frame = self._apply_ranking(
    selected_frame,
    spec,
)
diagnostics["pre_ranking_count"] = pre_ranking_count
diagnostics["post_ranking_eligible_count"] = len(selected_frame)
diagnostics["ranking_field"] = (
    spec.ranking.field
    if spec.ranking is not None
    else None
)
```

- [ ] **Step 6: Run the universe suite**

```bash
uv run pytest tests/unit/universe -q
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add src/qmt_agent_trader/universe/resolver.py \
        tests/unit/universe/test_resolver.py
git commit -m "fix(universe): exclude incomplete liquidity rankings"
```

---

# Task 5: Require Non-Empty Valid Index Membership Evidence

**Files:**
- Modify: `src/qmt_agent_trader/universe/pit_metadata.py`
- Modify: `src/qmt_agent_trader/universe/resolver.py`
- Modify: `tests/unit/universe/test_index_membership_asof.py`
- Modify: `tests/unit/universe/test_resolver.py`

**Interfaces:**
- Produces `_normalized_member_values(values: pd.Series, *, field: str) -> list[str]`.
- Membership maps include a code only when at least one valid normalized member exists.
- Invalid non-empty member values raise `INDEX_MEMBERSHIP_SOURCE_INVALID`.
- Resolver raises `INDEX_MEMBERSHIP_NOT_READY` for codes without current membership.

- [ ] **Step 1: Write the expired-history regression**

Append to `tests/unit/universe/test_index_membership_asof.py`:

```python
def test_index_with_only_expired_history_has_no_current_evidence() -> None:
    frame = pd.DataFrame(
        [
            {
                "index_code": "000905.SH",
                "con_code": "000001.SZ",
                "in_date": "20200101",
                "out_date": "20231231",
            }
        ]
    )

    observed = index_interval_members_by_code_asof(
        frame,
        ["000905.SH"],
        date(2024, 2, 15),
    )

    assert observed == {}
```

Import `index_interval_members_by_code_asof`.

- [ ] **Step 2: Write the malformed member regression**

```python
@pytest.mark.parametrize(
    "member",
    [None, "", "not-a-symbol"],
)
def test_latest_weight_snapshot_rejects_invalid_member(member) -> None:
    frame = pd.DataFrame(
        [
            {
                "index_code": "000300.SH",
                "con_code": member,
                "trade_date": "20240201",
            }
        ]
    )

    with pytest.raises(BacktestUniverseIntegrityError) as exc_info:
        index_weight_members_by_code_asof(
            frame,
            ["000300.SH"],
            date(2024, 2, 15),
        )

    assert exc_info.value.code == "INDEX_MEMBERSHIP_SOURCE_INVALID"
    assert exc_info.value.field == "raw/tushare/index_weight.con_code"
```

- [ ] **Step 3: Run the regressions**

```bash
uv run pytest \
  tests/unit/universe/test_index_membership_asof.py::test_index_with_only_expired_history_has_no_current_evidence \
  tests/unit/universe/test_index_membership_asof.py::test_latest_weight_snapshot_rejects_invalid_member -q
```

Expected: FAIL because expired codes currently map to empty lists and members are not validated.

- [ ] **Step 4: Add member normalization**

In `pit_metadata.py`, import:

```python
from qmt_agent_trader.universe.validators import normalize_symbol
```

Add:

```python
def _normalized_member_values(
    values: pd.Series,
    *,
    field: str,
) -> list[str]:
    normalized: list[str] = []
    invalid_count = 0
    for raw in values.tolist():
        text = "" if raw is None else str(raw).strip()
        symbol = normalize_symbol(text) if text else None
        if symbol is None:
            invalid_count += 1
            continue
        if symbol not in normalized:
            normalized.append(symbol)
    if invalid_count:
        raise BacktestUniverseIntegrityError(
            code="INDEX_MEMBERSHIP_SOURCE_INVALID",
            message="index membership contains invalid member identifiers",
            field=field,
            details={
                "invalid_row_count": invalid_count,
            },
        )
    return sorted(normalized)
```

- [ ] **Step 5: Validate weight and interval members**

Inside `index_weight_members_by_code_asof()`:

```python
members = _normalized_member_values(
    snapshot["con_code"],
    field="raw/tushare/index_weight.con_code",
)
if members:
    result[str(index_code)] = members
```

Replace the final interval dictionary comprehension with:

```python
result: dict[str, list[str]] = {}
for code in sorted(requested):
    code_active = active[
        active["index_code"].eq(code)
    ]
    if code_active.empty:
        continue
    members = _normalized_member_values(
        code_active["con_code"],
        field="raw/tushare/index_member.con_code",
    )
    if members:
        result[code] = members
return result
```

Remove `evidence_codes`.

- [ ] **Step 6: Make resolver reject empty memberships defensively**

Use:

```python
weight_members = weight_by_code.get(code) or []
interval_members = member_by_code.get(code) or []
if weight_members:
    members = weight_members
elif interval_members:
    members = interval_members
else:
    missing_codes.append(code)
    continue
```

- [ ] **Step 7: Add the resolver regression**

Append to `tests/unit/universe/test_resolver.py`:

```python
def test_expired_index_history_is_not_valid_asof_evidence(
    tmp_path,
) -> None:
    lake = DataLake(tmp_path / "lake", tmp_path / "research.duckdb")
    lake.write_parquet(
        pd.DataFrame(
            [
                {
                    "index_code": "000905.SH",
                    "con_code": "000001.SZ",
                    "in_date": "20200101",
                    "out_date": "20231231",
                }
            ]
        ),
        "raw",
        "tushare/index_member",
    )

    with pytest.raises(BacktestUniverseIntegrityError) as exc_info:
        UniverseResolver(lake)._index_constituents(
            ["000905.SH"],
            date(2024, 2, 15),
        )

    assert exc_info.value.code == "INDEX_MEMBERSHIP_NOT_READY"
    assert exc_info.value.details["missing_index_codes"] == [
        "000905.SH"
    ]
```

- [ ] **Step 8: Run focused tests**

```bash
uv run pytest \
  tests/unit/universe/test_index_membership_asof.py \
  tests/unit/universe/test_resolver.py -q
```

Expected: PASS.

- [ ] **Step 9: Commit**

```bash
git add src/qmt_agent_trader/universe/pit_metadata.py \
        src/qmt_agent_trader/universe/resolver.py \
        tests/unit/universe/test_index_membership_asof.py \
        tests/unit/universe/test_resolver.py
git commit -m "fix(universe): require active valid index members"
```

---

# Task 6: Migrate the Profiling Script and Invalidate Old Cache Semantics

**Files:**
- Modify: `scripts/profile_research_tools.py`
- Modify: `src/qmt_agent_trader/agent/tools/strategy_tools.py`
- Create: `tests/unit/scripts/test_profile_research_tools.py`
- Modify: `tests/unit/agent/test_backtest_cache_provenance.py`

**Interfaces:**
- Produces `_profile_strategy_spec(top_n: int) -> StrategySpec`.
- Produces `_profile_backtest_config(start: str, end: str, symbols: list[str]) -> StrategyBacktestConfig`.
- Profiling uses `strategy_identity_mode="adhoc"`.
- Engine semantic version becomes `2026-07-universe-session-integrity-v4`.

- [ ] **Step 1: Write the profiling-config test**

Create `tests/unit/scripts/test_profile_research_tools.py`:

```python
from scripts.profile_research_tools import (
    _profile_backtest_config,
)


def test_profile_backtest_config_uses_valid_adhoc_identity() -> None:
    config = _profile_backtest_config(
        start="20240101",
        end="20240201",
        symbols=["000001.SZ", "000002.SZ"],
    )

    assert config.strategy_identity_mode == "adhoc"
    assert config.strategy_spec is not None
    assert config.strategy_id == config.strategy_spec.strategy_id
    assert config.factor_name == "momentum_20d"
    assert config.strategy_spec.factors[0].factor_id == "momentum_20d"
    assert config.top_n == 2
```

- [ ] **Step 2: Run the test**

```bash
uv run pytest tests/unit/scripts/test_profile_research_tools.py -q
```

Expected: FAIL because `_profile_backtest_config()` does not exist.

- [ ] **Step 3: Add profiling helpers**

Import:

```python
from qmt_agent_trader.strategy.models import (
    StrategyKind,
    StrategySpec,
)
```

Add before `main()`:

```python
def _profile_strategy_spec(
    top_n: int,
) -> StrategySpec:
    return StrategySpec.model_validate(
        {
            "strategy_id": "profile_momentum_20d",
            "name": "Profile momentum 20d",
            "kind": StrategyKind.FACTOR_RANK_LONG_ONLY,
            "factors": [{"factor_id": "momentum_20d"}],
            "portfolio": {"top_n": top_n},
            "rebalance": {"frequency": "daily"},
        }
    )


def _profile_backtest_config(
    *,
    start: str,
    end: str,
    symbols: list[str],
) -> StrategyBacktestConfig:
    top_n = min(5, max(1, len(symbols)))
    spec = _profile_strategy_spec(top_n)
    return StrategyBacktestConfig(
        strategy_id=spec.strategy_id,
        strategy_identity_mode="adhoc",
        strategy_spec=spec,
        factor_name="momentum_20d",
        start_date=start,
        end_date=end,
        symbols=symbols,
        top_n=top_n,
        max_single_position_pct=spec.portfolio.max_single_position_pct,
        cash_buffer_pct=spec.portfolio.cash_buffer_pct,
        rebalance_frequency=spec.rebalance.frequency,
        min_turnover_threshold=spec.rebalance.min_turnover_threshold,
        rank_buffer=spec.rebalance.rank_buffer,
        execution_delay_days=spec.execution.execution_delay_days,
        slippage_bps=spec.execution.slippage_bps,
        lower_is_better=False,
    )
```

Replace the inline `StrategyBacktestConfig(...)` in `main()` with:

```python
_profile_backtest_config(
    start=start,
    end=end,
    symbols=symbols,
)
```

- [ ] **Step 4: Bump the engine semantic version**

In `strategy_tools.py`, set:

```python
BACKTEST_CACHE_SCHEMA_VERSION = "factor-rank-v4"
BACKTEST_ENGINE_SEMANTIC_VERSION = (
    "2026-07-universe-session-integrity-v4"
)
```

- [ ] **Step 5: Add the semantic-version regression**

Append to `tests/unit/agent/test_backtest_cache_provenance.py`:

```python
def test_provenance_records_universe_session_semantic_version(
    tmp_path,
) -> None:
    lake = DataLake(
        tmp_path / "lake",
        tmp_path / "research.duckdb",
    )
    spec = StrategySpec.model_validate(
        {
            "strategy_id": "adhoc_fixture",
            "name": "Adhoc fixture",
            "kind": "FACTOR_RANK_LONG_ONLY",
            "factors": [{"factor_id": "momentum_20d"}],
        }
    )
    config = StrategyBacktestConfig(
        strategy_id=spec.strategy_id,
        strategy_identity_mode="adhoc",
        strategy_spec=spec,
        factor_name="momentum_20d",
        start_date="20240101",
        end_date="20240131",
    )

    manifest = strategy_tools._backtest_provenance_manifest(
        lake,
        config=config,
        requested_factor_ids=["momentum_20d"],
        saved_strategy=None,
        effective_code_path=None,
        resolved_universe=None,
    )

    assert manifest["engine_semantic_version"] == (
        "2026-07-universe-session-integrity-v4"
    )
```

- [ ] **Step 6: Run focused tests**

```bash
uv run pytest \
  tests/unit/scripts/test_profile_research_tools.py \
  tests/unit/agent/test_backtest_cache_provenance.py \
  tests/unit/strategy/test_backtest_identity_mode.py -q
```

Expected: PASS.

- [ ] **Step 7: Run the profiling smoke command**

```bash
uv run python scripts/profile_research_tools.py --quick
```

Expected:

- exit code `0`;
- with no local market data, a structured `NO_LOCAL_DATA` payload;
- with valid local data, no Pydantic validation failure for `strategy_identity_mode` or `strategy_spec`.

- [ ] **Step 8: Commit**

```bash
git add scripts/profile_research_tools.py \
        src/qmt_agent_trader/agent/tools/strategy_tools.py \
        tests/unit/scripts/test_profile_research_tools.py \
        tests/unit/agent/test_backtest_cache_provenance.py
git commit -m "fix(tooling): migrate profiling to adhoc identity"
```

---

# Task 7: Update Documentation and Run Final Verification

**Files:**
- Modify: `docs/backtest/factor-rank-adapter.md`

**Interfaces:**
- Documentation matches the final session and universe contracts.
- No GitHub Actions changes.

- [ ] **Step 1: Document one effective-session boundary**

Add under `## Universe session authority`:

```markdown
A snapshot resolves exactly one effective market session. Bars, listing and
Delisting state, index membership, market-cap evidence, liquidity windows,
classification checks, and diagnostics all use that same session. The original
requested date is retained only as request-audit metadata.

For a closed boundary, the previous open session is valid only when every
natural date between that session and the requested boundary has `trade_cal`
evidence.
```

Change `Delisting` to lowercase `delisting` while applying the text.

- [ ] **Step 2: Document mixed-asset completeness**

Add:

```markdown
For a mixed stock-and-ETF universe, the exact effective session must contain at
least one validated row for each requested asset type. A stock-only or ETF-only
partial snapshot raises `UNIVERSE_MARKET_SESSION_NOT_READY`.
```

- [ ] **Step 3: Document ranking eligibility**

Extend `## Liquidity-window completeness`:

```markdown
Rows with incomplete 20-session liquidity evidence are excluded before
`avg_amount_20d` or `avg_volume_20d` ranking. They are never used to fill a
requested Top-N after valid candidates are exhausted.
```

- [ ] **Step 4: Document index evidence**

Extend `## Point-in-time date and index evidence`:

```markdown
Historical membership rows with no active member at the effective session are
not current evidence. A requested index code resolves only when at least one
valid normalized member exists. Invalid member identifiers fail closed with
`INDEX_MEMBERSHIP_SOURCE_INVALID`; no current membership raises
`INDEX_MEMBERSHIP_NOT_READY`.
```

- [ ] **Step 5: Run focused regressions**

```bash
uv run pytest \
  tests/unit/data/test_trading_calendar.py \
  tests/unit/universe/test_exact_session_resolution.py \
  tests/unit/universe/test_resolver.py \
  tests/unit/universe/test_index_membership_asof.py \
  tests/unit/universe/test_universe_resolver_rolling.py \
  tests/unit/scripts/test_profile_research_tools.py \
  tests/unit/agent/test_backtest_cache_provenance.py \
  tests/unit/strategy/test_backtest_identity_mode.py \
  tests/unit/agent/test_adhoc_factor_strategy_identity.py -q
```

Expected: PASS.

- [ ] **Step 6: Run affected subsystem suites**

```bash
uv run pytest \
  tests/unit/data \
  tests/unit/universe \
  tests/unit/backtest \
  tests/unit/strategy \
  tests/unit/persistence/test_dataset_manifests.py \
  tests/unit/agent/test_adhoc_factor_strategy_identity.py \
  tests/unit/agent/test_backtest_pre_cache_identity.py \
  tests/unit/agent/test_backtest_cache_provenance.py -q
```

Expected: PASS.

- [ ] **Step 7: Run the profiling smoke command**

```bash
uv run python scripts/profile_research_tools.py --quick
```

Expected: exit code `0`; no `StrategyBacktestConfig` validation error.

- [ ] **Step 8: Verify one effective-session resolution per snapshot**

```bash
rg -n "latest_open_session_on_or_before" \
  src/qmt_agent_trader/universe/resolver.py
```

Expected: one use in `_resolve_effective_session()` and no independent calls in `_load_recent_bars()`, `_avg_20d_metrics()`, `_market_cap_asof()`, or `_resolve_for_date()`.

- [ ] **Step 9: Verify PIT helpers receive effective dates**

```bash
rg -n "_index_constituents|_avg_20d_metrics|_market_cap_asof|security_master_asof" \
  src/qmt_agent_trader/universe/resolver.py
```

Every call must receive `session.effective_date` or a parameter derived directly from it.

- [ ] **Step 10: Verify incomplete liquidity cannot reach ranking**

```bash
rg -n "_ranking_eligible_rows|amount_observation_count|volume_observation_count" \
  src/qmt_agent_trader/universe/resolver.py \
  tests/unit/universe
```

Expected: ranking uses `_ranking_eligible_rows()` and both count fields have direct regressions.

- [ ] **Step 11: Verify cache semantics**

```bash
rg -n "BACKTEST_CACHE_SCHEMA_VERSION|BACKTEST_ENGINE_SEMANTIC_VERSION" \
  src/qmt_agent_trader/agent/tools/strategy_tools.py \
  tests/unit/agent/test_backtest_cache_provenance.py \
  docs/backtest/factor-rank-adapter.md
```

Expected:

```text
BACKTEST_CACHE_SCHEMA_VERSION = "factor-rank-v4"
BACKTEST_ENGINE_SEMANTIC_VERSION = "2026-07-universe-session-integrity-v4"
```

- [ ] **Step 12: Run repository gates**

```bash
make check
```

Expected: exit code `0`.

- [ ] **Step 13: Commit documentation**

```bash
git add docs/backtest/factor-rank-adapter.md
git commit -m "docs(backtest): document session-aligned universe integrity"
```

---

# Final Acceptance Checklist

## Effective session

- [ ] Snapshot resolves one effective market session exactly once.
- [ ] Bars use the effective session.
- [ ] Listing and delisting state use the effective session.
- [ ] Index membership uses the effective session.
- [ ] Market-cap evidence uses the effective session.
- [ ] Liquidity windows end at the effective session.
- [ ] Requested closed dates appear only as request-audit metadata.
- [ ] Diagnostics expose requested and effective dates separately.

## Calendar authority

- [ ] The request boundary has `trade_cal` evidence.
- [ ] Every natural date between the previous open session and boundary has evidence.
- [ ] Fully evidenced weekends and holidays resolve correctly.
- [ ] Intermediate calendar gaps raise `TRADING_CALENDAR_PARTIAL_COVERAGE`.
- [ ] Rolling rebalance dates continue to come only from `trade_cal`.

## Mixed-asset coverage

- [ ] Stock-only universe requires stock bars.
- [ ] ETF-only universe requires ETF bars.
- [ ] Mixed universe requires both stock and ETF bars.
- [ ] Missing requested asset types raise `UNIVERSE_MARKET_SESSION_NOT_READY`.
- [ ] Error details list requested, observed, and missing asset types.

## Liquidity ranking

- [ ] `avg_amount_20d` requires 20 sessions and 20 non-null amount observations.
- [ ] `avg_volume_20d` requires 20 sessions and 20 non-null volume observations.
- [ ] Incomplete rows remain unavailable.
- [ ] Incomplete rows are removed before liquidity ranking.
- [ ] Invalid rows never fill Top-N after valid rows are exhausted.
- [ ] Ranking diagnostics expose pre-ranking and eligible counts.

## Index membership

- [ ] Each requested index code resolves independently.
- [ ] Weight evidence uses the latest snapshot at or before the effective session.
- [ ] Interval evidence uses active intervals at the effective session.
- [ ] Expired historical membership is not current evidence.
- [ ] Empty current membership raises `INDEX_MEMBERSHIP_NOT_READY`.
- [ ] Invalid member identifiers raise `INDEX_MEMBERSHIP_SOURCE_INVALID`.
- [ ] At least one normalized member is required for success.

## Tooling and cache

- [ ] Profiling builds an inline ad-hoc `StrategySpec`.
- [ ] Profiling config uses `strategy_identity_mode="adhoc"`.
- [ ] Profiling smoke command does not fail Pydantic validation.
- [ ] Cache schema remains `factor-rank-v4`.
- [ ] Engine semantic version is `2026-07-universe-session-integrity-v4`.
- [ ] Old successful-result cache entries cannot survive the new universe semantics.

## Safety and verification

- [ ] `research_only=True`.
- [ ] `live_trading_allowed=False`.
- [ ] Integrity failures create no completed report.
- [ ] Integrity failures create no successful cache entry.
- [ ] Unexpected programming exceptions propagate.
- [ ] Focused tests pass.
- [ ] Affected subsystem suites pass.
- [ ] Profiling smoke command exits successfully.
- [ ] `make check` passes.
- [ ] Documentation matches implementation.

## Explicitly Out of Scope

- Historical industry/theme classification implementation.
- A new full DataLake-to-Agent integration suite.
- GitHub Actions creation or modification.
- Historical extreme-drawdown reproduction.
- Process-isolated generated-strategy Python execution.
- New ETF-specific fundamental classification sources.

## Expected Merge Decision

Keep `REQUEST_CHANGES` until every acceptance item and local verification command passes. After implementation, perform one final static branch review against the new head before merging.
