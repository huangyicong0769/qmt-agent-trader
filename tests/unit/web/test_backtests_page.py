from pathlib import Path

from qmt_agent_trader.web.ui.pages.backtests import _equity_echart, _parse_report


def _report():
    return _parse_report(
        Path("research_1.json"),
        {
            "schema_version": "2.0",
            "run_id": "research_1",
            "artifact_type": "strategy_backtest",
            "metrics": {
                "total_return": -0.12,
                "sharpe": -0.5,
                "max_drawdown": -0.20,
                "turnover": 0.30,
                "trade_count": 12,
            },
            "diagnostics": {"status": "WARN", "checks": []},
            "equity_points": [
                {"trade_date": "2024-01-02", "equity": 100_000},
                {"trade_date": "2024-01-03", "equity": 88_000},
            ],
            "trade_blotter": [],
        },
    )


def test_parse_schema_v2_strategy_report() -> None:
    report = _report()
    assert report.total_return == -0.12
    assert report.sharpe == -0.5
    assert report.max_drawdown == -0.20
    assert report.turnover == 0.30
    assert report.fills == 12
    assert report.status == "WARN"


def test_equity_chart_uses_real_equity_points() -> None:
    option = _equity_echart(_report())
    assert option is not None
    assert option["xAxis"]["data"] == ["2024-01-02", "2024-01-03"]
    assert option["series"][0]["data"] == [100000.0, 88000.0]
