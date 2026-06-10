import json

from qmt_agent_trader.services.research_report_service import (
    compare_research_reports,
    save_research_report,
)


def test_save_research_report_marks_artifact_as_research_only(tmp_path) -> None:
    reports_dir = tmp_path / "reports" / "research"

    receipt = save_research_report(
        reports_dir,
        artifact_type="factor_rank_sensitivity",
        title="Factor-rank sensitivity: momentum_20d",
        payload={
            "status": "completed",
            "summary": {"scenario_count": 2, "pass_ratio": 1.0},
            "runs": [],
        },
        metadata={"factor_name": "momentum_20d"},
        agent_notes="stable enough for deeper paper review",
        infrastructure_requests=["add capacity stress checks"],
    )

    path = reports_dir / f"{receipt['run_id']}.json"
    record = json.loads(path.read_text(encoding="utf-8"))

    assert receipt["status"] == "saved"
    assert record["research_only"] is True
    assert record["approval_status"] == "NOT_REQUESTED"
    assert record["live_trading_allowed"] is False
    assert record["decision_boundary"]["can_approve_strategy"] is False
    assert record["summary"]["scenario_count"] == 2


def test_compare_research_reports_returns_compact_summaries(tmp_path) -> None:
    reports_dir = tmp_path / "reports" / "research"
    save_research_report(
        reports_dir,
        artifact_type="factor_rank_sensitivity",
        title="Factor-rank sensitivity: volatility_20d",
        payload={"summary": {"scenario_count": 1, "pass_ratio": 0.0}},
        infrastructure_requests=["add borrow/liquidity diagnostics"],
    )

    compared = compare_research_reports(reports_dir, limit=5)

    assert compared["status"] == "compared"
    assert len(compared["runs"]) == 1
    assert compared["runs"][0]["approval_status"] == "NOT_REQUESTED"
    assert compared["infrastructure_requests"] == ["add borrow/liquidity diagnostics"]
