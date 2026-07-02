from __future__ import annotations

from qmt_agent_trader.agent.evidence_ledger import EvidenceLedger


def test_evidence_ledger_records_completed_but_failed_backtest_conflict() -> None:
    ledger = EvidenceLedger(run_id="run-ledger")

    ledger.record_tool_result(
        "run_backtest",
        {
            "execution_status": "OK",
            "domain_status": "FAILED",
            "evidence_status": "INVALID",
            "recommendation_status": "DO_NOT_RECOMMEND",
            "raw_status": "completed",
            "diagnostic_status": "FAIL",
            "diagnostics": {"status": "FAIL"},
        },
    )

    report = ledger.report()

    assert report["summary"]["invalid_count"] == 1
    assert any(
        item["type"] == "COMPLETED_WITH_FAILED_DIAGNOSTICS"
        for item in report["conflicts"]
    )


def test_final_answer_conflict_report_preserves_raw_answer() -> None:
    ledger = EvidenceLedger(run_id="run-final")
    ledger.record_tool_result(
        "run_backtest",
        {
            "execution_status": "OK",
            "domain_status": "FAILED",
            "evidence_status": "INVALID",
            "recommendation_status": "DO_NOT_RECOMMEND",
            "raw_status": "completed",
            "diagnostic_status": "FAIL",
        },
    )
    raw = "低换手+低波动是最有希望的策略方向。"

    report = ledger.final_answer_conflict_report(raw)

    assert report["final_answer_raw"] == raw
    assert report["has_conflict"] is True
    assert report["severity"] == "HIGH"
    assert any(
        item["type"] == "UNSUPPORTED_RECOMMENDATION"
        for item in report["conflicts"]
    )
