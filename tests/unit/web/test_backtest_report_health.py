from pathlib import Path

from qmt_agent_trader.web.ui.pages.backtests import ReportCollection, _load_reports


def test_backtest_page_surfaces_degraded_excluded_report(tmp_path: Path) -> None:
    report = tmp_path / "backtests/bt_corrupt.json"
    report.parent.mkdir(parents=True)
    report.write_text("{broken")

    reports = _load_reports(tmp_path)

    assert isinstance(reports, ReportCollection)
    assert reports.storage_status == "DEGRADED"
    assert len(reports.excluded_reasons) == 1
    assert "bt_corrupt.json" in reports.excluded_reasons[0]
