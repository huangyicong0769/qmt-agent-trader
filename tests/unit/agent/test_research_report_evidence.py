from __future__ import annotations

import json
from pathlib import Path

from qmt_agent_trader.agent.schemas import ToolContext
from qmt_agent_trader.agent.tools.strategy_tools import generate_research_report_tool
from qmt_agent_trader.core.config import Settings


def test_research_report_does_not_promote_failed_or_unverified_candidates(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "qmt_agent_trader.agent.tools.strategy_tools.get_settings",
        lambda: Settings(project_root=tmp_path),
    )
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


def test_research_report_returns_structured_evidence_summary(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "qmt_agent_trader.agent.tools.strategy_tools.get_settings",
        lambda: Settings(project_root=tmp_path),
    )
    reports = tmp_path / "reports" / "research"
    reports.mkdir(parents=True)
    (reports / "run_strategy.json").write_text(
        json.dumps(
            {
                "run_id": "run_strategy",
                "artifact_type": "strategy_backtest",
                "strategy_id": "strat_a",
                "factor_ids": ["factor_a", "momentum_20d"],
                "execution_backend": "factor_rank_composite_adapter",
                "factor_weights": {"factor_a": 0.6, "momentum_20d": 0.4},
                "research_only": True,
                "live_trading_allowed": False,
                "metrics": {
                    "total_return": -0.12,
                    "sharpe": 0.01,
                    "max_drawdown": -0.28,
                },
                "diagnostics": {"status": "FAIL", "checks": []},
                "data_window": {
                    "requested_end": "2026-06-26",
                    "actual_end": "20260626",
                    "data_freshness": "covers_requested_end",
                },
                "config": {
                    "symbols": ["000001.SZ", "600519.SH"],
                    "start_date": "2024-03-01",
                    "end_date": "2026-06-26",
                    "universe": "custom_2_stocks",
                },
            }
        ),
        encoding="utf-8",
    )

    result = generate_research_report_tool.run(
        {"run_ids": ["run_strategy"]},
        ToolContext(run_id="report-evidence-summary", session_id="chat_report"),
    )

    text = Path(result["report_path"]).read_text(encoding="utf-8")
    summary = result["evidence_summary"][0]
    assert "status=completed, diagnostics=FAIL" in text
    assert summary["status"] == "completed"
    assert summary["diagnostics_status"] == "FAIL"
    assert summary["metrics"]["total_return"] == -0.12
    assert summary["strategy_id"] == "strat_a"
    assert summary["symbols"] == ["000001.SZ", "600519.SH"]
