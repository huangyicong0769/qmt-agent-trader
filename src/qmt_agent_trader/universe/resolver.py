"""Point-in-time universe resolver for snapshot and rolling modes."""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any

import pandas as pd

from qmt_agent_trader.backtest.errors import BacktestUniverseIntegrityError
from qmt_agent_trader.data.bars import load_daily_bars, normalize_tushare_daily
from qmt_agent_trader.data.integrity import require_unique_symbol_dates
from qmt_agent_trader.data.storage import DataLake
from qmt_agent_trader.data.trading_calendar import (
    latest_open_session_on_or_before,
    load_session_window,
    open_sessions_between,
)
from qmt_agent_trader.universe.fingerprints import fingerprint_spec, fingerprint_symbols
from qmt_agent_trader.universe.models import UniverseRule, UniverseSpec
from qmt_agent_trader.universe.pit_metadata import (
    index_interval_members_by_code_asof,
    index_weight_members_by_code_asof,
    require_historical_classification_support,
    security_master_asof,
)
from qmt_agent_trader.universe.validators import normalize_symbol

LIQUIDITY_WINDOW_SESSIONS = 20


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


def _field_evidence_eligible_rows(
    frame: pd.DataFrame,
    field: str,
) -> pd.DataFrame:
    if field not in frame.columns:
        return frame.iloc[0:0].copy()
    count_field = {
        "avg_amount_20d": "amount_observation_count",
        "avg_volume_20d": "volume_observation_count",
    }.get(field)
    if count_field is None:
        return frame.dropna(subset=[field]).copy()
    if count_field not in frame.columns:
        return frame.iloc[0:0].copy()

    complete_evidence = pd.Series(
        (
            _has_complete_liquidity_evidence(value, count)
            for value, count in zip(
                frame[field],
                frame[count_field],
                strict=True,
            )
        ),
        index=frame.index,
        dtype=bool,
    )
    return frame.loc[complete_evidence].copy()


def _has_complete_liquidity_evidence(
    value: Any,
    observation_count: Any,
) -> bool:
    numeric_value = _float_or_none(value)
    numeric_count = _float_or_none(observation_count)
    return bool(
        numeric_value is not None
        and math.isfinite(numeric_value)
        and numeric_count == float(LIQUIDITY_WINDOW_SESSIONS)
    )


def _requires_stock_master(
    spec: UniverseSpec,
) -> bool:
    if "stock" in spec.asset_types:
        return True
    return spec.selection.mode in {
        "industry",
        "theme",
    }


class UniverseResolver:
    def __init__(self, lake: DataLake) -> None:
        self.lake = lake

    def build(
        self,
        spec: UniverseSpec,
        *,
        as_of_date: str | None = None,
        mode: str | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
        rebalance_frequency: str | None = None,
        limit: int | None = None,
        include_exclusions: bool = False,
    ) -> dict[str, Any]:
        requested_mode = mode or spec.mode
        if requested_mode == "snapshot":
            resolve_date = _date_key(as_of_date or end_date or _latest_available_date(self.lake))
            return self._build_snapshot(
                spec,
                as_of_date=resolve_date,
                limit=limit,
                include_exclusions=include_exclusions,
            )
        if requested_mode == "rolling":
            if not start_date or not end_date:
                return {
                    "status": "INVALID_REQUEST",
                    "reason": "ROLLING_REQUIRES_START_AND_END_DATE",
                    "rolling_symbols": {},
                    "metadata": {"empty_dates": []},
                }
            return self._build_rolling(
                spec,
                start_date=_date_key(start_date),
                end_date=_date_key(end_date),
                rebalance_frequency=rebalance_frequency or spec.rebalance_frequency,
                limit=limit,
                include_exclusions=include_exclusions,
            )
        return {
            "status": "INVALID_REQUEST",
            "reason": "UNSUPPORTED_UNIVERSE_MODE",
            "allowed_modes": ["snapshot", "rolling"],
        }

    def _build_snapshot(
        self,
        spec: UniverseSpec,
        *,
        as_of_date: str,
        limit: int | None,
        include_exclusions: bool,
    ) -> dict[str, Any]:
        symbols, excluded, diagnostics = self._resolve_for_date(spec, as_of_date=as_of_date)
        symbols, limit_metadata = _apply_limit(symbols, spec=spec, limit=limit)
        diagnostics["selected_count"] = len(symbols)
        metadata: dict[str, Any] = {
            "count": len(symbols),
            "as_of_date": as_of_date,
            "fingerprint": fingerprint_symbols(spec, mode="snapshot", symbols=symbols),
            "spec_fingerprint": fingerprint_spec(spec),
            "excluded_symbols": excluded if include_exclusions else [],
            "diagnostics": diagnostics,
            "candidate_count": int(diagnostics.get("candidate_count", len(symbols))),
            **limit_metadata,
        }
        return {
            "status": "OK",
            "mode": "snapshot",
            "symbols": symbols,
            "metadata": metadata,
        }

    def _build_rolling(
        self,
        spec: UniverseSpec,
        *,
        start_date: str,
        end_date: str,
        rebalance_frequency: str,
        limit: int | None,
        include_exclusions: bool,
    ) -> dict[str, Any]:
        resolve_dates = self._rebalance_dates(
            spec,
            start_date=start_date,
            end_date=end_date,
            frequency=rebalance_frequency,
        )
        rolling_symbols: dict[str, list[str]] = {}
        excluded_by_date: dict[str, list[dict[str, str]]] = {}
        diagnostics_by_date: dict[str, dict[str, Any]] = {}
        limit_metadata_by_date: dict[str, dict[str, object]] = {}
        for resolve_date in resolve_dates:
            symbols, excluded, diagnostics = self._resolve_for_date(spec, as_of_date=resolve_date)
            resolved, limit_metadata = _apply_limit(symbols, spec=spec, limit=limit)
            rolling_symbols[resolve_date] = resolved
            diagnostics["selected_count"] = len(resolved)
            diagnostics_by_date[resolve_date] = diagnostics
            limit_metadata_by_date[resolve_date] = limit_metadata
            if include_exclusions:
                excluded_by_date[resolve_date] = excluded

        counts = [len(symbols) for symbols in rolling_symbols.values()]
        empty_dates = [item for item, symbols in rolling_symbols.items() if not symbols]
        changed_dates = 0
        previous: list[str] | None = None
        for symbols in rolling_symbols.values():
            if previous is not None and symbols != previous:
                changed_dates += 1
            previous = symbols
        metadata: dict[str, Any] = {
            "resolve_dates": list(rolling_symbols),
            "min_count": min(counts) if counts else 0,
            "max_count": max(counts) if counts else 0,
            "mean_count": sum(counts) / len(counts) if counts else 0.0,
            "fingerprint": fingerprint_symbols(
                spec,
                mode="rolling",
                rolling_symbols=rolling_symbols,
            ),
            "spec_fingerprint": fingerprint_spec(spec),
            "empty_dates": empty_dates,
            "changed_dates": changed_dates,
            "diagnostics_by_date": diagnostics_by_date,
            "limit_metadata_by_date": limit_metadata_by_date,
            "truncated": any(bool(item["truncated"]) for item in limit_metadata_by_date.values()),
            "effective_limit": spec.max_symbols if spec.max_symbols is not None else limit,
            "truncation_source": (
                "spec.max_symbols"
                if spec.max_symbols is not None
                else "request_limit"
                if limit is not None
                else None
            ),
            "pre_limit_selected_count": max(
                (
                    value if isinstance(value := item["pre_limit_selected_count"], int) else 0
                    for item in limit_metadata_by_date.values()
                ),
                default=0,
            ),
            "selected_count": max(counts, default=0),
            "candidate_count": max(
                (
                    value if isinstance(value := item.get("candidate_count"), int) else 0
                    for item in diagnostics_by_date.values()
                ),
                default=0,
            ),
        }
        if include_exclusions:
            metadata["excluded_symbols_by_date"] = excluded_by_date
        return {
            "status": "OK",
            "mode": "rolling",
            "rolling_symbols": rolling_symbols,
            "metadata": metadata,
        }

    def _resolve_for_date(
        self,
        spec: UniverseSpec,
        *,
        as_of_date: str,
    ) -> tuple[list[str], list[dict[str, str]], dict[str, Any]]:
        session = _resolve_effective_session(
            self.lake,
            as_of_date,
        )
        recent = self._load_recent_bars(
            session.effective_date,
            spec.asset_types,
        )
        stock_basic = (
            security_master_asof(
                self._stock_basic(),
                session.effective_date,
            )
            if _requires_stock_master(spec)
            else pd.DataFrame(
                columns=[
                    "symbol",
                    "display_name",
                    "list_date",
                    "delist_date",
                    "listed_as_of",
                ]
            )
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
        selected: list[dict[str, Any]] = []
        excluded: list[dict[str, str]] = []
        for row in candidates.to_dict(orient="records"):
            symbol = str(row.get("symbol", ""))
            reason = self._exclusion_reason(
                spec,
                row,
                as_of_date=session.effective_key,
            )
            if reason is not None:
                excluded.append({"symbol": symbol, "reason": reason})
                continue
            selected.append(row)
        selected_frame = pd.DataFrame(selected)
        selected_frame = self._apply_rules(selected_frame, spec.selection.rules)
        pre_ranking_count = len(selected_frame)
        selected_frame = self._apply_ranking(selected_frame, spec)
        diagnostics = _universe_diagnostics(
            recent=recent,
            stock_basic=stock_basic,
            candidates=candidates,
            as_of_date=session.effective_key,
            selection_mode=spec.selection.mode,
            selected_count=0 if selected_frame.empty else len(selected_frame),
            candidate_count=candidate_count,
            excluded=excluded,
        )
        diagnostics["requested_as_of_date"] = session.requested_key
        diagnostics["effective_market_session"] = session.effective_key
        diagnostics["pre_ranking_count"] = pre_ranking_count
        diagnostics["post_ranking_eligible_count"] = len(selected_frame)
        diagnostics["ranking_field"] = (
            spec.ranking.field if spec.ranking is not None else None
        )
        if selected_frame.empty:
            return [], excluded, diagnostics
        return _ordered_unique_symbols(selected_frame, spec), excluded, diagnostics

    def _select_candidates(
        self,
        spec: UniverseSpec,
        recent: pd.DataFrame,
        stock_basic: pd.DataFrame,
        *,
        effective_date: date,
    ) -> pd.DataFrame:
        selection = spec.selection
        if selection.mode == "explicit_symbols":
            return _candidate_frame_for_symbols(selection.symbols, recent, stock_basic)
        if selection.mode == "industry":
            stock_matches = stock_basic[
                stock_basic.get("industry", pd.Series(dtype=object))
                .astype(str)
                .isin(selection.industries)
            ]
            return _candidate_frame_for_symbols(
                stock_matches.get("ts_code", pd.Series(dtype=object)).astype(str).tolist(),
                recent,
                stock_basic,
            )
        if selection.mode == "theme":
            return self._theme_candidates(selection.theme_concepts, recent, stock_basic)
        if selection.mode == "index_constituents":
            return _candidate_frame_for_symbols(
                self._index_constituents(selection.index_codes, effective_date),
                recent,
                stock_basic,
            )
        if selection.mode == "etf_category":
            return self._etf_category_candidates(selection.theme_concepts, recent)
        return _merge_recent_and_stock_basic(recent, stock_basic)

    def _theme_candidates(
        self,
        concepts: list[str],
        recent: pd.DataFrame,
        stock_basic: pd.DataFrame,
    ) -> pd.DataFrame:
        if stock_basic.empty or not concepts:
            return _merge_recent_and_stock_basic(recent, stock_basic).iloc[0:0].copy()
        haystack = (
            stock_basic.get("name", pd.Series(dtype=object)).astype(str)
            + " "
            + stock_basic.get("industry", pd.Series(dtype=object)).astype(str)
        )
        mask = pd.Series(False, index=stock_basic.index)
        for concept in concepts:
            mask = mask | haystack.str.contains(concept, case=False, regex=False, na=False)
        symbols = stock_basic.loc[mask, "ts_code"].astype(str).tolist()
        return _candidate_frame_for_symbols(symbols, recent, stock_basic)

    def _etf_category_candidates(
        self,
        categories: list[str],
        recent: pd.DataFrame,
    ) -> pd.DataFrame:
        if not categories:
            raise BacktestUniverseIntegrityError(
                code="UNIVERSE_PIT_CLASSIFICATION_NOT_READY",
                message="ETF category selection lacks category values",
                field="classification_history",
                details={
                    "selection_mode": "etf_category",
                },
            )
        raise BacktestUniverseIntegrityError(
            code="UNIVERSE_PIT_CLASSIFICATION_NOT_READY",
            message=(
                "historical ETF category selection requires dated "
                "classification evidence"
            ),
            field="classification_history",
            details={
                "selection_mode": "etf_category",
                "categories": list(categories),
            },
        )

    def _exclusion_reason(
        self,
        spec: UniverseSpec,
        row: dict[str, Any],
        *,
        as_of_date: str,
    ) -> str | None:
        symbol = str(row.get("symbol", ""))
        if not symbol:
            return "missing_symbol"
        filters = spec.filters
        if filters.require_bar_coverage and not bool(row.get("has_bar_coverage", False)):
            return "no_bar_coverage"
        if str(row.get("asset_type") or "stock") == "stock":
            listed_as_of = row.get("listed_as_of")
            if _is_missing_scalar(listed_as_of) or not bool(listed_as_of):
                return "not_listed_as_of"
            list_date_raw = row.get("list_date")
            if not _is_missing_scalar(list_date_raw):
                listed_days = (
                    _parse_date(as_of_date) - _parse_date(str(list_date_raw))
                ).days
            if listed_days < 0:
                return "not_yet_listed"
            if listed_days < filters.min_listed_days:
                return "listed_days_below_minimum"
        if filters.exclude_st and bool(row["st"]):
            return "st"
        if filters.exclude_suspended and bool(row["suspended"]):
            return "suspended"
        if filters.min_avg_amount_20d is not None:
            amount_count = _float_or_none(row.get("amount_observation_count"))
            if amount_count != float(LIQUIDITY_WINDOW_SESSIONS):
                return "amount_20d_coverage_incomplete"
        if filters.min_avg_volume_20d is not None:
            volume_count = _float_or_none(row.get("volume_observation_count"))
            if volume_count != float(LIQUIDITY_WINDOW_SESSIONS):
                return "volume_20d_coverage_incomplete"
        if (
            filters.min_avg_amount_20d is not None
            and not _has_complete_liquidity_evidence(
                row.get("avg_amount_20d"),
                row.get("amount_observation_count"),
            )
        ):
            return "amount_coverage_missing"
        if filters.min_avg_amount_20d is not None and float(row.get("avg_amount_20d") or 0) < float(
            filters.min_avg_amount_20d
        ):
            return "avg_amount_20d_below_minimum"
        if (
            filters.min_avg_volume_20d is not None
            and not _has_complete_liquidity_evidence(
                row.get("avg_volume_20d"),
                row.get("volume_observation_count"),
            )
        ):
            return "volume_coverage_missing"
        if filters.min_avg_volume_20d is not None and float(row.get("avg_volume_20d") or 0) < float(
            filters.min_avg_volume_20d
        ):
            return "avg_volume_20d_below_minimum"
        if filters.require_fundamental_coverage and _float_or_none(row.get("market_cap")) is None:
            return "fundamental_coverage_missing"
        market_cap = _float_or_none(row.get("market_cap"))
        if filters.min_market_cap is not None:
            if market_cap is None:
                return "market_cap_missing"
            if market_cap < filters.min_market_cap:
                return "market_cap_below_minimum"
        if filters.max_market_cap is not None:
            if market_cap is None:
                return "market_cap_missing"
            if market_cap > filters.max_market_cap:
                return "market_cap_above_maximum"
        return None

    def _apply_rules(
        self,
        frame: pd.DataFrame,
        rules: list[UniverseRule],
    ) -> pd.DataFrame:
        if frame.empty or not rules:
            return frame

        filtered = frame.copy()
        for rule in rules:
            filtered = _field_evidence_eligible_rows(
                filtered,
                rule.field,
            )
            if filtered.empty:
                return filtered
            series = filtered[rule.field]
            mask = _rule_mask(
                series,
                rule,
            ).fillna(False)
            filtered = filtered.loc[mask].copy()
        return filtered

    def _apply_ranking(self, frame: pd.DataFrame, spec: UniverseSpec) -> pd.DataFrame:
        ranking = spec.ranking
        if frame.empty or ranking is None:
            return frame
        if ranking.field not in frame.columns or "symbol" not in frame.columns:
            return frame.iloc[0:0].copy()
        eligible = _field_evidence_eligible_rows(
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

    def _attach_metrics(
        self,
        candidates: pd.DataFrame,
        spec: UniverseSpec,
        *,
        effective_date: date,
    ) -> pd.DataFrame:
        if candidates.empty:
            return candidates
        data = candidates.copy()
        needs_20d = (
            spec.filters.min_avg_amount_20d is not None
            or spec.filters.min_avg_volume_20d is not None
            or (
                spec.ranking is not None
                and spec.ranking.field in {"avg_amount_20d", "avg_volume_20d"}
            )
            or any(
                rule.field in {"avg_amount_20d", "avg_volume_20d"} for rule in spec.selection.rules
            )
        )
        if needs_20d:
            metrics = self._avg_20d_metrics(effective_date, spec.asset_types)
            data = data.merge(metrics, on="symbol", how="left")
        needs_market_cap = (
            spec.filters.require_fundamental_coverage
            or spec.filters.min_market_cap is not None
            or spec.filters.max_market_cap is not None
            or (spec.ranking is not None and spec.ranking.field == "market_cap")
            or any(rule.field == "market_cap" for rule in spec.selection.rules)
        )
        if needs_market_cap:
            data = data.merge(self._market_cap_asof(effective_date), on="symbol", how="left")
        return data

    def _avg_20d_metrics(
        self,
        effective_date: date,
        asset_types: Sequence[str],
    ) -> pd.DataFrame:
        end_key = f"{effective_date:%Y%m%d}"
        window = load_session_window(
            self.lake,
            start=end_key,
            end=end_key,
            warmup_sessions=19,
        )
        start_key = f"{window.panel_start:%Y%m%d}"
        frames: list[pd.DataFrame] = []
        for dataset, asset_type in _datasets_for_asset_types(asset_types):
            path = self.lake.dataset_path("raw", dataset)
            if not path.exists():
                continue
            raw = self.lake.read_parquet_filtered(
                "raw",
                dataset,
                start=start_key,
                end=end_key,
                columns=["ts_code", "trade_date", "open", "high", "low", "close", "vol", "amount"],
            )
            if raw.empty:
                continue
            normalized = normalize_tushare_daily(raw, asset_type=asset_type)
            frames.append(normalized)
        if not frames:
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
        bars = pd.concat(frames, ignore_index=True).sort_values(["symbol", "trade_date"])
        allowed_dates = set((*window.warmup_dates, *window.expected_dates))
        bars = bars[bars["trade_date"].isin(allowed_dates)]
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

    def _market_cap_asof(self, effective_date: date) -> pd.DataFrame:
        path = self.lake.dataset_path("raw", "tushare/daily_basic")
        if not path.exists():
            return pd.DataFrame(columns=["symbol", "market_cap"])
        raw = self.lake.read_parquet_filtered(
            "raw",
            "tushare/daily_basic",
            end=f"{effective_date:%Y%m%d}",
            columns=["ts_code", "trade_date", "total_mv"],
        )
        if raw.empty or "total_mv" not in raw.columns:
            return pd.DataFrame(columns=["symbol", "market_cap"])
        raw = raw.rename(columns={"ts_code": "symbol", "total_mv": "market_cap"})
        raw["trade_date"] = pd.to_datetime(raw["trade_date"].astype(str), format="%Y%m%d").dt.date
        require_unique_symbol_dates(
            raw,
            symbol_column="symbol",
            date_column="trade_date",
            code="DUPLICATE_UNIVERSE_SOURCE_KEY",
            field="raw/tushare/daily_basic",
        )
        return (
            raw.sort_values(["symbol", "trade_date"])
            .groupby("symbol", as_index=False)
            .tail(1)[["symbol", "market_cap"]]
        )

    def _load_recent_bars(
        self,
        effective_date: date,
        asset_types: Sequence[str],
    ) -> pd.DataFrame:
        key = f"{effective_date:%Y%m%d}"
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
        requested_asset_types = {str(item) for item in asset_types}
        observed_asset_types = {
            str(item) for item in exact["asset_type"].dropna().astype(str)
        }
        missing_asset_types = requested_asset_types.difference(observed_asset_types)
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
                    "requested_asset_types": sorted(requested_asset_types),
                    "observed_asset_types": sorted(observed_asset_types),
                    "missing_asset_types": sorted(missing_asset_types),
                },
            )
        return exact

    def _stock_basic(self) -> pd.DataFrame:
        path = self.lake.dataset_path("raw", "tushare/stock_basic")
        if not path.exists():
            return pd.DataFrame()
        return self.lake.read_parquet("raw", "tushare/stock_basic")

    def _index_constituents(
        self,
        index_codes: list[str],
        effective_date: date,
    ) -> list[str]:
        normalized_codes = list(dict.fromkeys(str(code) for code in index_codes))
        weight_by_code: dict[str, list[str]] = {}
        member_by_code: dict[str, list[str]] = {}

        weight_path = self.lake.dataset_path("raw", "tushare/index_weight")
        if weight_path.exists():
            weight_by_code = index_weight_members_by_code_asof(
                self.lake.read_parquet("raw", "tushare/index_weight"),
                normalized_codes,
                effective_date,
            )

        member_path = self.lake.dataset_path("raw", "tushare/index_member")
        if member_path.exists():
            member_by_code = index_interval_members_by_code_asof(
                self.lake.read_parquet("raw", "tushare/index_member"),
                normalized_codes,
                effective_date,
            )

        missing_codes: list[str] = []
        ordered_members: list[str] = []
        for code in normalized_codes:
            weight_members = weight_by_code.get(code) or []
            interval_members = member_by_code.get(code) or []
            if weight_members:
                members = weight_members
            elif interval_members:
                members = interval_members
            else:
                missing_codes.append(code)
                continue
            ordered_members.extend(members)

        if missing_codes:
            raise BacktestUniverseIntegrityError(
                code="INDEX_MEMBERSHIP_NOT_READY",
                message="one or more requested indices lack as-of membership evidence",
                trade_date=effective_date.isoformat(),
                field="index_membership",
                details={"missing_index_codes": missing_codes},
            )

        return [
            normalized
            for item in dict.fromkeys(ordered_members)
            if (normalized := normalize_symbol(item)) is not None
        ]

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


def _candidate_frame_for_symbols(
    symbols: Iterable[str],
    recent: pd.DataFrame,
    stock_basic: pd.DataFrame,
) -> pd.DataFrame:
    normalized = [symbol for symbol in (normalize_symbol(item) for item in symbols) if symbol]
    base = pd.DataFrame({"symbol": list(dict.fromkeys(normalized))})
    if base.empty:
        return pd.DataFrame(columns=["symbol"])
    recent_columns = [
        column
        for column in ("symbol", "trade_date", "asset_type", "st", "suspended", "volume", "amount")
        if column in recent.columns
    ]
    data = base.merge(recent[recent_columns], on="symbol", how="left")
    data["has_bar_coverage"] = data["trade_date"].notna() if "trade_date" in data.columns else False
    return _merge_stock_basic_columns(data, stock_basic)


def _merge_recent_and_stock_basic(recent: pd.DataFrame, stock_basic: pd.DataFrame) -> pd.DataFrame:
    if recent.empty:
        return pd.DataFrame(columns=["symbol", "has_bar_coverage"])
    data = recent.copy()
    data["has_bar_coverage"] = True
    return _merge_stock_basic_columns(data, stock_basic)


def _merge_stock_basic_columns(frame: pd.DataFrame, stock_basic: pd.DataFrame) -> pd.DataFrame:
    if stock_basic.empty or "symbol" not in stock_basic.columns:
        return frame
    columns = [
        column
        for column in (
            "symbol",
            "display_name",
            "list_date",
            "delist_date",
            "listed_as_of",
        )
        if column in stock_basic.columns
    ]
    basics = stock_basic[columns]
    return frame.merge(basics, on="symbol", how="left")


def _universe_diagnostics(
    *,
    recent: pd.DataFrame,
    stock_basic: pd.DataFrame,
    candidates: pd.DataFrame,
    as_of_date: str,
    selection_mode: str,
    selected_count: int,
    candidate_count: int,
    excluded: list[dict[str, str]],
) -> dict[str, Any]:
    latest_global_trade_date = None
    symbols_on_latest = 0
    symbols_with_bar = 0
    stale_symbol_count = 0
    max_staleness_days = 0
    if not recent.empty and "trade_date" in recent.columns and "symbol" in recent.columns:
        dates = pd.to_datetime(recent["trade_date"], errors="coerce")
        latest = dates.max()
        if pd.notna(latest):
            latest_global_trade_date = f"{latest.date():%Y%m%d}"
            symbols_on_latest = int(recent.loc[dates == latest, "symbol"].nunique())
        as_of = _parse_date(as_of_date)
        staleness = (pd.Timestamp(as_of) - dates).dt.days
        stale_symbol_count = int(recent.loc[staleness > 0, "symbol"].nunique())
        max_staleness_days = int(staleness.max()) if not staleness.empty else 0
        symbols_with_bar = int(recent["symbol"].nunique())
    stock_basic_count = (
        int(stock_basic["symbol"].astype(str).nunique())
        if not stock_basic.empty and "symbol" in stock_basic.columns
        else 0
    )
    exclusion_counts: dict[str, int] = {}
    for item in excluded:
        reason = item.get("reason", "unknown")
        exclusion_counts[reason] = exclusion_counts.get(reason, 0) + 1
    return {
        "as_of_date": as_of_date,
        "candidate_count": candidate_count,
        "selected_count": selected_count,
        "recent_bar_symbol_count": symbols_with_bar,
        "stock_basic_symbol_count": stock_basic_count,
        "latest_global_trade_date": latest_global_trade_date,
        "symbols_on_latest_global_trade_date": symbols_on_latest,
        "symbols_with_bar_before_as_of": symbols_with_bar,
        "stale_symbol_count": stale_symbol_count,
        "max_bar_staleness_days": max_staleness_days,
        "selection_mode": selection_mode,
        "excluded_count": len(excluded),
        "exclusion_counts": exclusion_counts,
        "candidate_columns": sorted(str(column) for column in candidates.columns),
    }


def _rule_mask(series: pd.Series, rule: UniverseRule) -> pd.Series:
    operator = rule.operator
    value = rule.value
    if operator == "eq":
        return series == value
    if operator == "ne":
        return series != value
    if operator == "in":
        values = value if isinstance(value, list) else [value]
        return series.isin(values)
    if operator == "not_in":
        values = value if isinstance(value, list) else [value]
        return ~series.isin(values)
    if operator in {"gt", "gte", "lt", "lte", "between"}:
        numeric = pd.to_numeric(series, errors="coerce")
        if operator == "gt":
            return numeric > float(value)
        if operator == "gte":
            return numeric >= float(value)
        if operator == "lt":
            return numeric < float(value)
        if operator == "lte":
            return numeric <= float(value)
        bounds = value if isinstance(value, list) else []
        if len(bounds) != 2:
            return pd.Series(False, index=series.index)
        return (numeric >= float(bounds[0])) & (numeric <= float(bounds[1]))
    text = series.astype(str)
    needle = str(value)
    if operator == "contains":
        return text.str.contains(needle, case=False, regex=False, na=False)
    if operator == "starts_with":
        return text.str.startswith(needle, na=False)
    if operator == "ends_with":
        return text.str.endswith(needle, na=False)
    return pd.Series(False, index=series.index)


def _is_missing_scalar(value: Any) -> bool:
    if value is None:
        return True
    try:
        if pd.isna(value):
            return True
    except (TypeError, ValueError):
        pass
    text = str(value).strip().lower()
    return text in {"", "nan", "nat", "none", "<na>"}


def _ordered_unique_symbols(frame: pd.DataFrame, spec: UniverseSpec) -> list[str]:
    if frame.empty or "symbol" not in frame.columns:
        return []
    ordered = frame
    if spec.ranking is None and spec.selection.mode != "explicit_symbols":
        ordered = frame.sort_values("symbol", kind="stable")
    return list(dict.fromkeys(ordered["symbol"].astype(str).tolist()))


def _apply_limit(
    symbols: list[str],
    *,
    spec: UniverseSpec,
    limit: int | None,
) -> tuple[list[str], dict[str, object]]:
    requested_limit = limit
    effective_limit = spec.max_symbols if spec.max_symbols is not None else requested_limit
    selected = symbols if effective_limit is None else symbols[:effective_limit]
    return selected, {
        "pre_limit_selected_count": len(symbols),
        "selected_count": len(selected),
        "truncated": len(selected) < len(symbols),
        "effective_limit": effective_limit,
        "truncation_source": (
            "spec.max_symbols"
            if spec.max_symbols is not None
            else "request_limit"
            if requested_limit is not None
            else None
        ),
    }


def _datasets_for_asset_types(asset_types: Sequence[str]) -> list[tuple[str, str]]:
    result: list[tuple[str, str]] = []
    if "stock" in asset_types:
        result.append(("tushare/daily", "stock"))
    if "etf" in asset_types:
        result.append(("tushare/fund_daily", "etf"))
    return result


def _trade_dates(lake: DataLake, asset_types: Sequence[str], *, start: str, end: str) -> list[str]:
    frames: list[pd.DataFrame] = []
    for dataset, _asset_type in _datasets_for_asset_types(asset_types):
        path = lake.dataset_path("raw", dataset)
        if not path.exists():
            continue
        raw = lake.read_parquet_filtered(
            "raw",
            dataset,
            columns=["trade_date"],
            start=start,
            end=end,
        )
        if not raw.empty and "trade_date" in raw.columns:
            frames.append(raw)
    if not frames:
        return []
    values = pd.concat(frames, ignore_index=True)["trade_date"].astype(str).tolist()
    return sorted({_date_key(value) for value in values})


def _latest_available_date(lake: DataLake) -> str:
    dates = _trade_dates(lake, ["stock", "etf"], start="19000101", end="29991231")
    if not dates:
        return datetime.now().strftime("%Y%m%d")
    return dates[-1]


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _date_key(value: str) -> str:
    return _parse_date(value).strftime("%Y%m%d")


def _parse_date(value: str) -> date:
    text = str(value)
    for fmt in ("%Y%m%d", "%Y-%m-%d"):
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    return datetime.fromisoformat(text).date()
