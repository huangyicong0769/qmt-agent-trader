from __future__ import annotations

import json
from pathlib import Path

from qmt_agent_trader.agent.schemas import ToolContext
from qmt_agent_trader.agent.tools.strategy_tools import generate_research_report_tool


def test_research_report_does_not_promote_failed_or_unverified_candidates(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    reports = tmp_path / "reports" / "research"
    reports.mkdir(parents=True)
    (reports / "run_failed.json").write_text(
        json.dumps(
            {
                "run_id": "run_failed",
                "status": "completed",
                "diagnostics": {"status": "FAIL", "checks": []},
                "candidate_type": "strategy",
                "universe_requested": "named_universe",
                "universe_effective": "default_universe",
                "symbols_source": "default_universe",
                "symbols_count": 5000,
                "generated_code": False,
                "static_checks": "NOT_RUN",
                "saved_in_registry": False,
                "execution_backend": "factor_rank_composite_adapter",
                "factor_weights": {"factor_a": 0.7, "factor_b": 0.3},
                "research_only": True,
                "live_trading_allowed": False,
                "warnings": ["diagnostics failed"],
            }
        ),
        encoding="utf-8",
    )

    result = generate_research_report_tool.run(
        {
            "experiment_id": "exp_evidence",
            "run_ids": ["run_failed"],
            "include_sections": ["summary", "metrics", "limitations"],
        },
        ToolContext(run_id="report-evidence", experiment_id="exp_evidence"),
    )

    text = Path(result["report_path"]).read_text(encoding="utf-8")
    effective_section = text.split("## Failed Candidates / 失败候选", 1)[0]
    assert "## Effective Candidates / 有效候选" in effective_section
    assert "- None" in effective_section
    assert "generated_code=False" in text
    assert "static_checks=NOT_RUN" in text
    assert "saved_in_registry=False" in text
    assert "symbols_source=default_universe" in text
    assert "diagnostics failed" in text
