"""Backtests page — interactive visualization of historical backtest results.

Lists all backtest/research reports with key metrics.
Click any report to drill into full detail: performance charts,
diagnostic checks, trade blotter, and sensitivity analysis.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from nicegui import ui

from qmt_agent_trader.web.ui.layout import shell

REPORTS_ROOT = Path("reports")


# ── Data model ──


@dataclass
class ReportMeta:
    """Lightweight summary of a single report."""
    path: Path
    run_id: str
    created_at: str
    title: str = ""
    artifact_type: str = ""
    symbol: str = ""
    total_return: float | None = None
    sharpe: float | None = None
    max_drawdown: float | None = None
    turnover: float | None = None
    fills: int = 0
    status: str = "PASS"
    checks: list[dict[str, Any]] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict)


def _load_reports() -> list[ReportMeta]:
    """Scan reports/backtests/ and reports/research/ for JSON reports."""
    results: list[ReportMeta] = []
    for glob_pattern in ("backtests/bt_*.json", "research/research_*.json"):
        for p in sorted(
            REPORTS_ROOT.glob(glob_pattern),
            key=lambda x: x.stat().st_mtime,
            reverse=True,
        ):
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                continue
            meta = _parse_report(p, data)
            results.append(meta)
    return results


def _parse_report(path: Path, data: dict[str, Any]) -> ReportMeta:
    """Extract structured metadata from a raw report dict."""
    payload = data.get("payload", {}) if isinstance(data.get("payload"), dict) else {}
    summary = data.get("summary", {}) if isinstance(data.get("summary"), dict) else {}

    # Metrics — try payload first, then summary
    metrics = payload if any(k in payload for k in ("total_return", "sharpe")) else summary
    total_return = metrics.get("total_return")
    sharpe = metrics.get("sharpe")
    max_dd = metrics.get("max_drawdown")
    turnover = metrics.get("turnover")

    # Symbol from holdings or trade_blotter
    holdings = data.get("holdings_report", {})
    if isinstance(holdings, dict):
        symbol = str(holdings.get("symbol", ""))
    else:
        symbol = ""
    if not symbol:
        blotter = data.get("trade_blotter", [])
        if isinstance(blotter, list) and blotter:
            symbol = str(blotter[0].get("symbol", "") if isinstance(blotter[0], dict) else "")

    # Fills
    perf = data.get("performance_report", {})
    fills = perf.get("fills", 0) if isinstance(perf, dict) else 0

    # Diagnostics
    diag = data.get("diagnostic_report", {})
    if isinstance(diag, dict):
        status = str(diag.get("status", "UNKNOWN"))
        checks = diag.get("checks", []) if isinstance(diag.get("checks"), list) else []
    else:
        rgate = data.get("review_gate", {})
        if not isinstance(rgate, dict):
            rgate = {}
        status = str(rgate.get("status", "UNKNOWN"))
        checks = rgate.get("checks", [])
        if not isinstance(checks, list):
            checks = []

    return ReportMeta(
        path=path,
        run_id=str(data.get("run_id", path.stem)),
        created_at=str(data.get("created_at", "")),
        title=str(data.get("title", "")),
        artifact_type=str(data.get("artifact_type", "")),
        symbol=symbol,
        total_return=float(total_return) if total_return is not None else None,
        sharpe=float(sharpe) if sharpe is not None else None,
        max_drawdown=float(max_dd) if max_dd is not None else None,
        turnover=float(turnover) if turnover is not None else None,
        fills=int(fills),
        status=status,
        checks=checks,
        raw=data,
    )


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


# ── Detail: equity curve chart ──


def _equity_echart(report: ReportMeta) -> dict[str, Any] | None:
    """Build an ECharts option dict for the equity curve.

    If the report doesn't contain equity data, derive a simple
    cumulative return line from trade dates and total return.
    """
    blotter = report.raw.get("trade_blotter", [])
    if not isinstance(blotter, list) or not blotter:
        return None

    dates: list[str] = []
    trades = 0
    for t in blotter:
        if isinstance(t, dict):
            d = t.get("execution_date", "")
            if d:
                dates.append(str(d))
                trades += 1

    if not dates or report.total_return is None:
        return None

    # Simple equity curve: linear interpolation from 1.0 to 1.0 + total_return
    n = len(dates)
    eq = [round(1.0 + report.total_return * (i / max(n - 1, 1)), 4) for i in range(n)]

    return {
        "tooltip": {"trigger": "axis"},
        "xAxis": {"type": "category", "data": dates, "axisLabel": {"rotate": 45, "fontSize": 10}},
        "yAxis": {
            "type": "value",
            "name": "Equity",
            "axisLabel": {"formatter": "{value:.2f}"},
        },
        "series": [
            {
                "name": "Equity",
                "type": "line",
                "data": eq,
                "smooth": True,
                "areaStyle": {"color": "rgba(59,130,246,0.15)"},
                "lineStyle": {"color": "#3b82f6", "width": 2},
                "itemStyle": {"color": "#3b82f6"},
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
    """Build an ECharts option dict for sensitivity scenarios."""
    payload = report.raw.get("payload", {})
    if not isinstance(payload, dict):
        return None
    scenario_results = payload.get("scenario_results", payload.get("runs", []))
    if not isinstance(scenario_results, list) or len(scenario_results) < 2:
        return None

    scenarios: list[str] = []
    returns: list[float] = []
    for sr in scenario_results:
        if isinstance(sr, dict):
            label = str(sr.get("label", sr.get("scenario", "")) or "")
            if not label:
                continue
            scenarios.append(label)
            metrics = sr.get("metrics", {})
            if isinstance(metrics, dict):
                returns.append(float(metrics.get("total_return", 0)))
            else:
                returns.append(0.0)

    if not scenarios:
        return None

    return {
        "tooltip": {"trigger": "axis", "axisPointer": {"type": "shadow"}},
        "xAxis": {
            "type": "category",
            "data": scenarios,
            "axisLabel": {"rotate": 30, "fontSize": 9, "width": 80, "overflow": "truncate"},
        },
        "yAxis": {
            "type": "value",
            "name": "Return",
            "axisLabel": {"formatter": "{value:.1%}"},
        },
        "series": [
            {
                "name": "Total Return",
                "type": "bar",
                "data": returns,
                "itemStyle": {
                    "color": {
                        "type": "linear",
                        "x": 0, "y": 0, "x2": 0, "y2": 1,
                        "colorStops": [
                            {"offset": 0, "color": "#3b82f6"},
                            {"offset": 1, "color": "#93c5fd"},
                        ],
                    },
                    "borderRadius": [4, 4, 0, 0],
                },
            }
        ],
        "grid": {"left": 60, "right": 20, "top": 20, "bottom": 80},
    }


# ── Page ──


def register() -> None:
    @ui.page("/backtests")
    def backtests_page() -> None:
        shell("Backtests")

        reports = _load_reports()
        detail_ref: dict[str, Any] = {"visible": False, "report": None}
        list_container: dict[str, ui.column | None] = {"c": None}
        detail_container: dict[str, ui.column | None] = {"c": None}

        def show_list() -> None:
            dc = detail_container["c"]
            lc = list_container["c"]
            if dc:
                dc.set_visibility(False)
            if lc:
                lc.set_visibility(True)
            detail_ref["visible"] = False

        def show_detail(report: ReportMeta) -> None:
            detail_ref["report"] = report
            detail_ref["visible"] = True
            dc = detail_container["c"]
            lc = list_container["c"]
            if lc:
                lc.set_visibility(False)
            if dc:
                dc.clear()
                with dc:
                    _render_detail(report, show_list)
                dc.set_visibility(True)

        with ui.column().classes("w-full gap-4"):
            # ── Header ──
            with ui.row().classes("w-full items-center justify-between"):
                ui.label("Backtests").classes("text-2xl font-semibold")
                ui.button(
                    "← Back to list",
                    on_click=show_list,
                ).props("flat").bind_visibility_from(
                    detail_ref, "visible"
                )

            # ── Report list ──
            list_col = ui.column().classes("w-full gap-3")
            list_container["c"] = list_col
            with list_col:
                _render_list(reports, show_detail)

            # ── Report detail ──
            detail_col = ui.column().classes("w-full gap-4")
            detail_col.set_visibility(False)
            detail_container["c"] = detail_col


# ── List view ──


def _render_list(
    reports: list[ReportMeta],
    on_click: Any,
) -> None:
    if not reports:
        ui.label("No backtest reports found.").classes("text-gray-500 p-8")
        return

    with ui.row().classes("w-full gap-3 flex-wrap"):
        for r in reports:
            _report_card(r, on_click)


def _report_card(r: ReportMeta, on_click: Any) -> None:
    with ui.card().classes(
        "w-72 cursor-pointer hover:shadow-md transition-shadow"
    ).on("click", lambda r=r: on_click(r)):
        with ui.column().classes("gap-2 w-full"):
            # Header row
            with ui.row().classes("w-full items-center justify-between"):
                ui.label(
                    r.title or r.symbol or r.run_id[-8:]
                ).classes("text-sm font-semibold truncate")
                ui.chip(
                    r.status, icon=_status_icon(r.status), color=_status_color(r.status)
                ).props("size=sm dense")

            # Artifact type
            atype = r.artifact_type or (
                "Backtest" if "bt_" in r.run_id else "Research"
            )
            ui.label(atype).classes("text-xs text-gray-400")

            # Metrics grid
            if r.total_return is not None:
                with ui.row().classes("w-full gap-3"):
                    with ui.column().classes("gap-0"):
                        ret_cls = (
                            "text-green-600"
                            if (r.total_return or 0) >= 0
                            else "text-red-600"
                        )
                        ui.label(_fmt_pct(r.total_return)).classes(
                            f"text-lg font-bold {ret_cls}"
                        )
                        ui.label("Return").classes("text-[10px] text-gray-400")
                    with ui.column().classes("gap-0"):
                        ui.label(_fmt_num(r.sharpe)).classes("text-lg font-bold")
                        ui.label("Sharpe").classes("text-[10px] text-gray-400")
                    with ui.column().classes("gap-0"):
                        ui.label(_fmt_pct(r.max_drawdown)).classes(
                            "text-lg font-bold text-red-500"
                        )
                        ui.label("Max DD").classes("text-[10px] text-gray-400")

            # Footer: date + symbol
            with ui.row().classes("w-full justify-between text-[10px] text-gray-400"):
                created = r.created_at[:10] if r.created_at else ""
                ui.label(created or "")
                if r.symbol:
                    ui.label(r.symbol)

            # Checks summary
            passed = sum(1 for c in r.checks if c.get("status") == "PASS")
            failed = sum(1 for c in r.checks if c.get("status") == "FAIL")
            warn = sum(1 for c in r.checks if c.get("status") == "WARN")
            if r.checks:
                with ui.row().classes("gap-2"):
                    if passed:
                        ui.chip(f"{passed} ✓", color="green").props("size=xs dense")
                    if warn:
                        ui.chip(f"{warn} ⚠", color="orange").props("size=xs dense")
                    if failed:
                        ui.chip(f"{failed} ✗", color="red").props("size=xs dense")


# ── Detail view ──


def _render_detail(r: ReportMeta, on_back: Any) -> None:
    atype = r.artifact_type or ("Backtest" if "bt_" in r.run_id else "Research")

    with ui.column().classes("w-full gap-6"):
        # ── Title bar ──
        with ui.row().classes("w-full items-center gap-3"):
            ui.label(r.title or r.run_id).classes("text-xl font-semibold")
            ui.chip(atype, color="blue").props("size=sm dense")
            ui.chip(
                r.status, icon=_status_icon(r.status), color=_status_color(r.status)
            ).props("size=sm dense")

        with ui.row().classes("w-full gap-2 text-sm text-gray-500"):
            ui.label(f"Run: {r.run_id}")
            ui.label("·")
            ui.label(f"Created: {r.created_at[:19] or '—'}")
            if r.symbol:
                ui.label("·")
                ui.label(f"Symbol: {r.symbol}")

        # ── Performance metrics cards ──
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
            _metric_card("Turnover", _fmt_pct(r.turnover), "swap_horiz", "")
            _metric_card("Fills", str(r.fills), "receipt_long", "")

        # ── Equity curve chart ──
        eq_option = _equity_echart(r)
        if eq_option:
            with ui.card().classes("w-full p-4"):
                ui.label("Equity Curve").classes("text-sm font-semibold mb-2")
                ui.echart(eq_option).classes("w-full").style("height: 320px")

        # ── Sensitivity chart ──
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
                                _status_icon(s), color=_status_color(s), size="sm"
                            )
                            with ui.column().classes("gap-0 flex-1"):
                                ui.label(name).classes("text-sm font-medium")
                                if msg:
                                    ui.label(msg).classes("text-xs text-gray-500")
                            with ui.row().classes("gap-2 text-xs"):
                                if obs is not None:
                                    obs_str = (
                                        f"{obs:+.2%}"
                                        if isinstance(obs, float) and abs(obs) < 10
                                        else str(obs)
                                    )
                                    ui.label(f"Observed: {obs_str}").classes("text-gray-500")
                                if thr is not None:
                                    thr_str = (
                                        f"{thr:+.2%}"
                                        if isinstance(thr, float) and abs(thr) < 10
                                        else str(thr)
                                    )
                                    ui.label(f"Threshold: {thr_str}").classes("text-gray-400")

        # ── Trade blotter ──
        blotter = r.raw.get("trade_blotter", [])
        if isinstance(blotter, list) and blotter:
            with ui.card().classes("w-full p-4"):
                ui.label(f"Trade Blotter ({len(blotter)} trades)").classes(
                    "text-sm font-semibold mb-2"
                )
                cols = [
                    {"name": "symbol", "label": "Symbol", "field": "symbol"},
                    {"name": "date", "label": "Execution Date", "field": "execution_date"},
                    {"name": "qty", "label": "Qty", "field": "quantity"},
                ]
                rows = [
                    {
                        "symbol": t.get("symbol", "") if isinstance(t, dict) else "",
                        "execution_date": (
                            t.get("execution_date", "")
                            if isinstance(t, dict)
                            else ""
                        ),
                        "quantity": t.get("quantity", "") if isinstance(t, dict) else "",
                    }
                    for t in blotter
                ]
                ui.table(columns=cols, rows=rows, row_key="execution_date").classes(
                    "w-full text-xs"
                )

        # ── Raw JSON (collapsible) ──
        with ui.expansion("Raw JSON", icon="code").classes("w-full"):
            ui.code(json.dumps(r.raw, ensure_ascii=False, indent=2, default=str)).classes(
                "w-full text-xs max-h-96 overflow-auto"
            )


def _metric_card(
    label: str, value: str, icon: str, value_class: str = ""
) -> None:
    with ui.card().classes("flex-1 min-w-[120px] p-4"):
        with ui.row().classes("items-center gap-2"):
            ui.icon(icon, size="sm").classes("text-gray-400")
            ui.label(label).classes("text-xs text-gray-500")
        ui.label(value).classes(f"text-2xl font-bold mt-1 {value_class}")
