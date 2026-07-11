"""Backtests page — interactive visualization of historical backtest results.

Lists all backtest/research reports with key metrics.
Click any report to drill into full detail: performance charts,
diagnostic checks, trade blotter, and sensitivity analysis.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from nicegui import ui

from qmt_agent_trader.core.config import get_settings
from qmt_agent_trader.persistence.artifacts import ArtifactManifest, artifact_store_for_root
from qmt_agent_trader.persistence.paths import PersistencePaths
from qmt_agent_trader.web.ui.layout import shell

logger = logging.getLogger(__name__)


# ── Data model ──


@dataclass
class ReportMeta:
    path: Path
    run_id: str
    created_at: str
    title: str = ""
    artifact_type: str = ""
    symbol: str = ""
    factor_name: str = ""
    total_return: float | None = None
    sharpe: float | None = None
    max_drawdown: float | None = None
    turnover: float | None = None
    fills: int = 0
    status: str = "UNKNOWN"
    checks: list[dict[str, Any]] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict)


def _load_reports(reports_root: Path | None = None) -> list[ReportMeta]:
    root = reports_root or PersistencePaths.from_settings(get_settings()).reports_root
    results: list[ReportMeta] = []
    for glob_pattern in ("backtests/bt_*.json", "research/research_*.json"):
        for p in sorted(
            root.glob(glob_pattern),
            key=lambda x: x.stat().st_mtime,
            reverse=True,
        ):
            try:
                store = artifact_store_for_root(p.parent)
                manifests = [
                    ArtifactManifest.model_validate_json(item.read_text(encoding="utf-8"))
                    for item in (p.parent / ".manifests").glob("*.json")
                ]
                manifest = next(item for item in manifests if item.relative_path == p.name)
                data = json.loads(
                    store.read_verified(manifest.artifact_id, expected_relative_path=p.name)
                )
            except Exception as exc:
                logger.warning("governed report excluded: %s (%s)", p, type(exc).__name__)
                continue
            results.append(_parse_report(p, data))
    return results


def _parse_report(path: Path, data: dict[str, Any]) -> ReportMeta:
    summary = data.get("summary", {})
    if not isinstance(summary, dict):
        summary = {}
    metadata = data.get("metadata", {})
    if not isinstance(metadata, dict):
        metadata = {}

    # ── Metrics ──
    # Research reports: summary.baseline_total_return
    # Smoke tests: performance_report.fills
    total_return = _extract_float(summary, "baseline_total_return")
    if total_return is None:
        total_return = _extract_float(summary, "median_total_return")

    # ── Symbol from metadata or trade blotter ──
    symbol = str(metadata.get("factor_name", "")) or ""
    if not symbol:
        holdings = data.get("holdings_report", {})
        if isinstance(holdings, dict):
            symbol = str(holdings.get("symbol", ""))
    if not symbol:
        blotter = data.get("trade_blotter", [])
        if isinstance(blotter, list) and blotter:
            first = blotter[0]
            if isinstance(first, dict):
                symbol = str(first.get("symbol", ""))

    # ── Fills ──
    perf = data.get("performance_report", {})
    fills = 0
    if isinstance(perf, dict):
        fills = int(perf.get("fills", 0))

    # ── Factor name (for title) ──
    factor_name = str(metadata.get("factor_name", "")) or ""

    # ── Diagnostics / review gate ──
    status = "UNKNOWN"
    checks: list[dict[str, Any]] = []

    # Try review_gate (research reports)
    rgate = data.get("review_gate", {})
    if isinstance(rgate, dict):
        st = str(rgate.get("status", "")).upper()
        status = _normalize_status(st)
        raw_checks = rgate.get("checks", [])
        if isinstance(raw_checks, list):
            checks = [_normalize_check(c) for c in raw_checks if isinstance(c, dict)]

    # Try diagnostic_report (some backtests)
    if status == "UNKNOWN":
        diag = data.get("diagnostic_report", {})
        if isinstance(diag, dict):
            st = str(diag.get("status", "")).upper()
            status = _normalize_status(st)
            raw_checks = diag.get("checks", [])
            if isinstance(raw_checks, list):
                checks = [_normalize_check(c) for c in raw_checks if isinstance(c, dict)]

    return ReportMeta(
        path=path,
        run_id=str(data.get("run_id", path.stem)),
        created_at=str(data.get("created_at", "")),
        title=str(data.get("title", "")),
        artifact_type=str(data.get("artifact_type", "")),
        symbol=symbol,
        factor_name=factor_name,
        total_return=total_return,
        sharpe=None,
        max_drawdown=None,
        turnover=None,
        fills=fills,
        status=status,
        checks=checks,
        raw=data,
    )


def _extract_float(d: dict[str, Any], key: str) -> float | None:
    v = d.get(key)
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _normalize_status(s: str) -> str:
    s_upper = s.upper().strip()
    if s_upper in ("PASS", "PASSED", "OK", "TRUE"):
        return "PASS"
    if s_upper in ("WARN", "WARNING", "WARNED"):
        return "WARN"
    if s_upper in ("FAIL", "FAILED", "ERROR", "FALSE"):
        return "FAIL"
    return s_upper if s_upper else "UNKNOWN"


def _normalize_check(c: dict[str, Any]) -> dict[str, Any]:
    st = _normalize_status(str(c.get("status", "")))
    return {
        "name": str(c.get("name", "")),
        "status": st,
        "observed": c.get("observed"),
        "threshold": c.get("threshold"),
        "message": str(c.get("message", "")),
    }


# ── Formatting helpers ──


def _fmt_pct(v: float | None) -> str:
    if v is None:
        return "—"
    return f"{v:+.2%}"


def _fmt_num(v: float | None, precision: int = 2) -> str:
    if v is None:
        return "—"
    return f"{v:.{precision}f}"


def _status_color(s: str) -> str:
    return {"PASS": "green", "WARN": "orange", "FAIL": "red"}.get(s, "grey")


def _status_icon(s: str) -> str:
    return {"PASS": "check_circle", "WARN": "warning", "FAIL": "cancel"}.get(
        s, "help"
    )


# ── ECharts helpers ──


def _equity_echart(report: ReportMeta) -> dict[str, Any] | None:
    blotter = report.raw.get("trade_blotter", [])
    if not isinstance(blotter, list) or not blotter:
        return None

    dates: list[str] = []
    for t in blotter:
        if isinstance(t, dict):
            d = t.get("execution_date", "")
            if d:
                dates.append(str(d))

    if not dates or report.total_return is None:
        return None

    n = len(dates)
    eq = [round(1.0 + report.total_return * (i / max(n - 1, 1)), 4) for i in range(n)]

    return {
        "tooltip": {"trigger": "axis"},
        "xAxis": {
            "type": "category",
            "data": dates,
            "axisLabel": {"rotate": 45, "fontSize": 10},
        },
        "yAxis": {
            "type": "value",
            "name": "Equity",
        },
        "series": [
            {
                "name": "Equity",
                "type": "line",
                "data": eq,
                "smooth": True,
                "areaStyle": {"color": "rgba(59,130,246,0.15)"},
                "lineStyle": {"color": "#3b82f6", "width": 2},
                "markLine": {
                    "silent": True,
                    "data": [
                        {
                            "yAxis": 1.0,
                            "label": {"formatter": "Baseline"},
                            "lineStyle": {"color": "#9ca3af", "type": "dashed"},
                        }
                    ],
                },
            }
        ],
        "grid": {"left": 60, "right": 30, "top": 20, "bottom": 50},
    }


def _sensitivity_echart(report: ReportMeta) -> dict[str, Any] | None:
    payload = report.raw.get("payload", {})
    if not isinstance(payload, dict):
        return None

    scenarios = payload.get("scenario_results", payload.get("runs", []))
    if not isinstance(scenarios, list) or len(scenarios) < 2:
        return None

    # Build data; sample to max 24 bars for readability
    items: list[tuple[str, float]] = []
    for sr in scenarios:
        if not isinstance(sr, dict):
            continue
        label = str(sr.get("label", sr.get("scenario", "")) or "")
        if not label:
            continue
        metrics = sr.get("metrics", {})
        if isinstance(metrics, dict):
            ret = float(metrics.get("total_return", 0))
        else:
            ret = 0.0
        items.append((label, ret))

    if len(items) > 24:
        step = max(1, len(items) // 24)
        items = items[::step]

    if not items:
        return None

    labels = [it[0] for it in items]
    returns = [it[1] for it in items]

    return {
        "tooltip": {
            "trigger": "axis",
            "axisPointer": {"type": "shadow"},
            "formatter": (
                        "function(params) {"
                        "  return params[0].name + '<br/>Return: ' +"
                        "    (params[0].value * 100).toFixed(2) + '%';"
                        "}"
                    ),
        },
        "xAxis": {
            "type": "category",
            "data": labels,
            "axisLabel": {"rotate": 60, "fontSize": 9, "interval": 0},
        },
        "yAxis": {
            "type": "value",
            "name": "Return",
            "axisLabel": {
                "formatter": "function(value) { return (value * 100).toFixed(0) + '%'; }",
            },
        },
        "series": [
            {
                "name": "Total Return",
                "type": "bar",
                "data": returns,
                "itemStyle": {
                    "color": (
                        "function(params) {"
                        "  return params.value >= 0 ? '#22c55e' : '#ef4444';"
                        "}"
                    ),
                    "borderRadius": [4, 4, 0, 0],
                },
            }
        ],
        "grid": {"left": 60, "right": 20, "top": 20, "bottom": 100},
    }


# ── Page ──


def register() -> None:
    @ui.page("/backtests")
    def backtests_page() -> None:
        shell("Backtests")

        reports = _load_reports()

        detail_ref: dict[str, Any] = {"visible": False, "report": None}
        list_container: ui.column | None = None
        detail_container: ui.column | None = None

        def show_list() -> None:
            nonlocal list_container, detail_container
            if detail_container:
                detail_container.set_visibility(False)
            if list_container:
                list_container.set_visibility(True)
            detail_ref["visible"] = False

        def show_detail(report: ReportMeta) -> None:
            nonlocal list_container, detail_container
            detail_ref["report"] = report
            detail_ref["visible"] = True
            if list_container:
                list_container.set_visibility(False)
            if detail_container:
                detail_container.clear()
                with detail_container:
                    _render_detail(report, show_list)
                detail_container.set_visibility(True)

        with ui.column().classes("w-full gap-4"):
            # ── Header ──
            with ui.row().classes("w-full items-center justify-between"):
                ui.button(
                    "← Back to list",
                    on_click=show_list,
                ).props("flat").bind_visibility_from(detail_ref, "visible")

            # ── Report list ──
            list_col = ui.column().classes("w-full gap-3")
            list_container = list_col
            with list_col:
                _render_list(reports, show_detail)

            # ── Report detail ──
            detail_col = ui.column().classes("w-full gap-4")
            detail_col.set_visibility(False)
            detail_container = detail_col


# ── List view ──


def _render_list(reports: list[ReportMeta], on_click: Any) -> None:
    if not reports:
        with ui.card().classes("w-full p-8"):
            ui.label("No backtest reports yet.").classes("text-gray-500")
            ui.label(
                "Run a backtest from the Chat page — "
                "they'll appear here automatically."
            ).classes("text-sm text-gray-400")
        return

    with ui.row().classes("w-full gap-3 flex-wrap"):
        for r in reports:
            _report_card(r, on_click)


def _report_card(r: ReportMeta, on_click: Any) -> None:
    # Build a friendly display name
    if r.factor_name:
        display_name = r.factor_name
    elif r.symbol:
        display_name = r.symbol
    elif r.title:
        display_name = r.title[:40]
    else:
        display_name = r.run_id[-12:]

    with ui.card().classes(
        "w-72 cursor-pointer hover:shadow-md transition-shadow border"
    ).on("click", lambda r=r: on_click(r)):
        with ui.column().classes("gap-2 w-full"):
            # Header
            with ui.row().classes("w-full items-center justify-between"):
                ui.label(display_name).classes("text-sm font-semibold truncate max-w-[160px]")
                color = _status_color(r.status)
                ui.chip(r.status, icon=_status_icon(r.status), color=color).props(
                    "size=sm dense"
                )

            # Type
            atype = r.artifact_type or ("Backtest" if "bt_" in r.run_id else "Research")
            ui.label(atype).classes("text-xs text-gray-400")

            # Metrics
            if r.total_return is not None:
                with ui.row().classes("w-full gap-4"):
                    ret_color = "text-green-600" if r.total_return >= 0 else "text-red-600"
                    ui.label(_fmt_pct(r.total_return)).classes(
                        f"text-xl font-bold {ret_color}"
                    )
                    ui.label("Return").classes("text-[10px] text-gray-400 self-end mb-0.5")

            # Footer
            created = r.created_at[:10] if r.created_at else ""
            footer_parts = [created]
            if r.symbol and r.symbol != display_name:
                footer_parts.append(r.symbol)
            # Checks summary
            if r.checks:
                passed = sum(1 for c in r.checks if c.get("status") == "PASS")
                failed = sum(1 for c in r.checks if c.get("status") == "FAIL")
                warn = sum(1 for c in r.checks if c.get("status") == "WARN")
                parts: list[str] = []
                if passed:
                    parts.append(f"{passed}✓")
                if warn:
                    parts.append(f"{warn}⚠")
                if failed:
                    parts.append(f"{failed}✗")
                if parts:
                    footer_parts.append(" ".join(parts))
            ui.label(" · ".join(f for f in footer_parts if f)).classes(
                "text-[10px] text-gray-400 w-full"
            )


# ── Detail view ──


def _render_detail(r: ReportMeta, on_back: Any) -> None:
    atype = r.artifact_type or ("Backtest" if "bt_" in r.run_id else "Research")
    display_name = (
        r.factor_name or r.symbol or r.title or r.run_id[-12:]
    )

    with ui.column().classes("w-full gap-6"):
        # ── Title bar ──
        with ui.row().classes("w-full items-center gap-3 flex-wrap"):
            ui.label(display_name).classes("text-xl font-semibold")
            ui.chip(atype, color="blue").props("size=sm dense")
            color = _status_color(r.status)
            ui.chip(
                r.status, icon=_status_icon(r.status), color=color
            ).props("size=sm dense")

        with ui.row().classes("w-full gap-2 text-sm text-gray-500 flex-wrap"):
            ui.label(f"Run: {r.run_id}")
            ui.label("·")
            ui.label(f"Created: {r.created_at[:19] or '—'}")
            if r.symbol:
                ui.label("·")
                ui.label(f"Symbol: {r.symbol}")

        # ── Performance metrics ──
        has_metrics = r.total_return is not None
        if has_metrics or r.fills > 0:
            with ui.row().classes("w-full gap-4 flex-wrap"):
                _metric_card(
                    "Total Return",
                    _fmt_pct(r.total_return),
                    "trending_up",
                    "text-green-600" if (r.total_return or 0) >= 0 else "text-red-600",
                )
                _metric_card("Sharpe", _fmt_num(r.sharpe), "speed", "")
                _metric_card(
                    "Max Drawdown",
                    _fmt_pct(r.max_drawdown),
                    "trending_down",
                    "text-red-500",
                )
                _metric_card(
                    "Turnover", _fmt_pct(r.turnover), "swap_horiz", ""
                )
                _metric_card("Trades", str(r.fills), "receipt_long", "")

        # ── Equity curve ──
        eq_option = _equity_echart(r)
        if eq_option:
            with ui.card().classes("w-full p-4"):
                ui.label("Equity Curve").classes("text-sm font-semibold mb-2")
                ui.echart(eq_option).classes("w-full").style("height: 320px")

        # ── Sensitivity ──
        sens_option = _sensitivity_echart(r)
        if sens_option:
            with ui.card().classes("w-full p-4"):
                ui.label("Sensitivity Analysis").classes("text-sm font-semibold mb-2")
                ui.echart(sens_option).classes("w-full").style("height: 350px")

        # ── Diagnostic checks ──
        if r.checks:
            with ui.card().classes("w-full p-4"):
                ui.label("Diagnostic Checks").classes("text-sm font-semibold mb-3")
                with ui.column().classes("w-full gap-2"):
                    for c in r.checks:
                        s = str(c.get("status", ""))
                        name = str(c.get("name", ""))
                        msg = str(c.get("message", ""))
                        obs = c.get("observed")
                        thr = c.get("threshold")

                        with ui.row().classes("w-full items-center gap-3"):
                            ui.icon(
                                _status_icon(s),
                                color=_status_color(s),
                                size="sm",
                            )
                            with ui.column().classes("gap-0 flex-1"):
                                ui.label(name).classes("text-sm font-medium")
                                if msg:
                                    ui.label(msg).classes("text-xs text-gray-500")
                            with ui.row().classes("gap-3 text-xs"):
                                if obs is not None:
                                    ui.label(
                                        f"Observed: {_fmt_check_val(obs)}"
                                    ).classes("text-gray-500 font-mono")
                                if thr is not None:
                                    ui.label(
                                        f"Threshold: {_fmt_check_val(thr)}"
                                    ).classes("text-gray-400 font-mono")

        # ── Trade blotter ──
        blotter = r.raw.get("trade_blotter", [])
        if isinstance(blotter, list) and blotter:
            with ui.card().classes("w-full p-4"):
                ui.label(f"Trade Blotter ({len(blotter)} trades)").classes(
                    "text-sm font-semibold mb-2"
                )
                cols = [
                    {"name": "symbol", "label": "Symbol", "field": "symbol"},
                    {
                        "name": "date",
                        "label": "Execution Date",
                        "field": "execution_date",
                    },
                    {"name": "qty", "label": "Qty", "field": "quantity"},
                ]
                rows = [
                    {
                        "symbol": (
                            t.get("symbol", "") if isinstance(t, dict) else ""
                        ),
                        "execution_date": (
                            t.get("execution_date", "")
                            if isinstance(t, dict)
                            else ""
                        ),
                        "quantity": (
                            t.get("quantity", "") if isinstance(t, dict) else ""
                        ),
                    }
                    for t in blotter
                ]
                ui.table(
                    columns=cols, rows=rows, row_key="execution_date"
                ).classes("w-full text-xs")

        # ── Raw JSON ──
        with ui.expansion("Raw JSON", icon="code").classes("w-full"):
            ui.code(
                json.dumps(r.raw, ensure_ascii=False, indent=2, default=str)
            ).classes("w-full text-xs max-h-96 overflow-auto")


def _fmt_check_val(v: Any) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, float):
        if abs(v) < 10:
            return f"{v:+.4f}"
        return f"{v:.2f}"
    return str(v)


def _metric_card(label: str, value: str, icon: str, value_class: str = "") -> None:
    with ui.card().classes("flex-1 min-w-[120px] p-4"):
        with ui.row().classes("items-center gap-2"):
            ui.icon(icon, size="sm").classes("text-gray-400")
            ui.label(label).classes("text-xs text-gray-500")
        ui.label(value).classes(f"text-2xl font-bold mt-1 {value_class}")
